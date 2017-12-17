#include "mpi.h"
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace std;

void KeyboardDataInit(double *m, double *v, int size) {
    for (int i = 0; i<size; i++) {
        for (int j = 0; j<size; j++) {
            cout << "a[" << i + 1 << "]" << '[' << j + 1 << "]=";
            cin >> m[i*size + j];
        }
        cout << "b[" << i + 1 << "]= ";
        cin >> v[i];
    }
}

void RandomDataInit(double *m, double *v, int size) {
    srand(time(0));
    for (int i = 0; i<size; i++) {
        double sum = 0;
        for (int j = 0; j<size; j++) {
            if (i != j) {
                m[i*size + j] = rand() % 18 - 9;
                sum += abs(m[i*size + j]);
            }
        }
        int sign;
        if (rand() > 16000)
            sign = 1;
        else
            sign = -1;
        m[i*size + i] = sign *sum*(rand() / 32767.0 + 1);
        v[i] = rand() % 18 - 9;
    }
}

void KnownDataInit(double *m, double *v, int size) {
    ifstream input;
    input.open("knownMatrixA.txt");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            input >> m[i * size + j];
        }
    }
    input.close();
    input.open("knownMatrixB.txt");
    for (int i = 0; i < size; ++i) {
        input >> v[i];
    }
    input.close();
}

void readFromFile(double *A, double *B, int *size) {
    ifstream input;
    input.open("knownMatrixA.txt");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            input >> m[i * size + j];
        }
    }
    input.close();
    input.open("knownMatrixB.txt");
    for (int i = 0; i < size; ++i) {
        input >> v[i];
    }
    input.close();
}


int main(int argc, char* argv[])
{

    double* pA;
    double* pB;
    double* pX;
    double* pProcRowsA;
    double* pProcRowsB;
    double* pProcTempX;
    int Size, RowNum, ProcNum, ProcRank;
    double Eps, Norm, MaxNorm;
    double Start, Finish, Dt;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (ProcRank == 0) {
        cout << "Input the dimension  ";
        cin >> Size;
    }
    cout << "Proc " << ProcRank << " waiting for input" << endl;
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cout << "Proc " << ProcRank << " Size broadcated" << endl;
    RowNum = Size / ProcNum;
    cout << "RowNum = " << RowNum << endl;
    Eps = 0.001;
    pX = new double[Size];
    pProcRowsA = new double[RowNum*Size];
    pProcRowsB = new double[RowNum];
    pProcTempX = new double[RowNum];
    if (ProcRank == 0) {
        pA = new double[Size*Size];
        pB = new double[Size];
        RandomDataInit(pA, pB, Size);

        //KeyboardDataInit(pA, pB, Size);
        //KnownDataInit(pA, pB, Size);
        for (int i = 0; i<Size; i++) {
            pX[i] = 0;
        }
    }
    MPI_Bcast(pX, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cout << "Proc " << ProcRank << " pX broadcasted" << endl;
    MPI_Scatter(pA, RowNum*Size, MPI_DOUBLE, pProcRowsA, RowNum*Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(pB, RowNum, MPI_DOUBLE, pProcRowsB, RowNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    Start = MPI_Wtime();

    do {
        for (int i = 0; i < RowNum; i++) {
            pProcTempX[i] = pProcRowsB[i];
            for (int j = 0; j < Size; j++) {
                if (ProcRank*RowNum + i != j)
                    pProcTempX[i] -= pProcRowsA[i*Size + j] * pX[j];
            }
            pProcTempX[i] /= pProcRowsA[ProcRank*RowNum + i + i * Size];
        }


        cout << "Proc " << ProcRank << " temp X = ";
        for (int i = 0; i < RowNum; i++) {
            cout << pProcTempX[i] << " ";
        }
        cout << endl;

        Norm = fabs(pX[ProcRank*RowNum] - pProcTempX[0]);
        for (int i = 0; i < RowNum; i++) {
            if (fabs(pX[ProcRank*RowNum + i] - pProcTempX[i]) > Norm)
                Norm = fabs(pX[ProcRank*RowNum + i] - pProcTempX[i]);
        }
        MPI_Reduce(&Norm, &MaxNorm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Bcast(&MaxNorm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Allgather(pProcTempX, RowNum, MPI_DOUBLE, pX, RowNum, MPI_DOUBLE, MPI_COMM_WORLD);
    } while (MaxNorm > Eps);
    Finish = MPI_Wtime();
    if (ProcRank == 0) {
        cout << endl << "time solution of system=" << (Finish - Start) << " sec." << endl;
    }

    if (ProcRank == 0) {
        delete[]pA;
        delete[]pB;
    }
    delete[]pX;
    delete[]pProcRowsA;
    delete[]pProcRowsB;
    delete[]pProcTempX;
    MPI_Finalize();

    //cin;
    return 0;
}