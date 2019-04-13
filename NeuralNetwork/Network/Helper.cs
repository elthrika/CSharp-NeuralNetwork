using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    static class Helper
    {
        private static Random random = new Random();

        public static double Random { get { return random.NextDouble(); } }
        public static int RandomInt(int max) => random.Next(max);

        public static double[] RandomVector(int size)
        {
            var vec = new double[size];
            for (int i = 0; i < size; i++)
            {
                double r = Random;
                vec[i] = r;
            }
            return vec;
        }

        public static double[,,] RandomMatrix(int sizeW, int sizeH, int sizeD)
        {
            double[,,] rval = new double[sizeD, sizeH, sizeW];

            for (int d = 0; d < sizeD; d++)
            {
                for (int h = 0; h < sizeH; h++)
                {
                    for (int w = 0; w < sizeW; w++)
                    {
                        rval[d, h, w] = Random;
                    }
                }
            }
            return rval;
        }

        public static double[,] RandomMatrix(int sizeW, int sizeH)
        {
            double[,] rval = new double[sizeH, sizeW];

            for (int h = 0; h < sizeH; h++)
            {
                for (int w = 0; w < sizeW; w++)
                {
                    rval[h, w] = Random;
                }
            }
            return rval;
        }

        public static (int D, int H, int W) GetDimensions(this double[,,] matrix)
        {
            return (matrix.GetLength(0), matrix.GetLength(1), matrix.GetLength(2));
        }

        public static (int H, int W) GetDimensions(this double[,] matrix)
        {
            return (matrix.GetLength(0), matrix.GetLength(1));
        }

        public static double[,,] Transpose3D(this double[,,] matrix, (int z, int y, int x) newDims)
        {
            if (newDims.x + newDims.y + newDims.z != 3) throw new ArgumentException();
            double[,,] rval = new double[matrix.GetLength(newDims.z), matrix.GetLength(newDims.y), matrix.GetLength(newDims.x)];
            int[] o = new int[3] { 0, 0, 0 };
            for (int z = 0; z < matrix.GetLength(newDims.z); z++)
            {
                o[newDims.z] = z;
                for (int y = 0; y < matrix.GetLength(newDims.y); y++)
                {
                    o[newDims.y] = y;
                    for (int x = 0; x < matrix.GetLength(newDims.x); x++)
                    {
                        o[newDims.x] = x;
                        rval[z, y, x] = matrix[o[0], o[1], o[2]];
                    }
                }
            }
            return rval;
        }

        public static double[,,] ExtractN(this List<double[,,]> list, int n)
        {
            double[,,] rval = new double[list.Count, list[0].GetLength(1), list[0].GetLength(2)];
            for (int d = 0; d < list.Count; d++)
            {
                for (int h = 0; h < list[d].GetLength(1); h++)
                {
                    for (int w = 0; w < list[d].GetLength(2); w++)
                    {
                        rval[d, h, w] = list[d][n, h, w];
                    }
                }
            }
            return rval;
        }

        public static double[,] ExtractDim(this double[,,] matrix, int dim, int n) //dim:0 depth, dim:1 height, dim:2 width
        {
            int dim1 = dim == 0 ? 1 : 0;
            int dim2 = dim == 2 ? 1 : 2;
            double[,] rval = new double[matrix.GetLength(dim1), matrix.GetLength(dim2)];

            int[] o = new int[3];
            o[dim] = n;

            for (int i = 0; i < matrix.GetLength(dim1); i++)
            {
                o[dim1] = i;
                for (int j = 0; j < matrix.GetLength(dim2); j++)
                {
                    o[dim2] = j;
                    rval[i, j] = matrix[o[0], o[1], o[2]];
                }
            }
            return rval;
        }

        public static double[,] CollapseDepth(this double[,,] matrix)
        {
            double[,] rval = new double[matrix.GetLength(1), matrix.GetLength(2)];

            for (int d = 0; d < matrix.GetLength(0); d++)
            {
                for (int h = 0; h < matrix.GetLength(1); h++)
                {
                    for (int w = 0; w < matrix.GetLength(2); w++)
                    {
                        rval[h, w] += matrix[d, h, w];
                    }
                }
            }

            return rval;
        }

        public static T[,,] AddXYPadding<T>(T[,,] matrix, int padding)
        {
            return AddXYPadding(matrix, padding, () => default(T));
        }

        public static T[,,] AddXYPadding<T>(T[,,] matrix, int padding, Func<T> defaultelementGenerator)
        {
            int width = matrix.GetLength(2);
            int height = matrix.GetLength(1);
            int depth = matrix.GetLength(0);
            int extdWidth = width + 2 * padding;
            int extdHeight = height + 2 * padding;
            var extendedMatrix = new T[depth, extdHeight, extdWidth];
            var previousNeurons = matrix.Cast<T>();
            for (int d = 0; d < depth; d++)
            {
                for (int h = 0; h < extdHeight; h++)
                {
                    for (int w = 0; w < extdWidth; w++)
                    {
                        if (w < padding || h < padding || w > width + padding - 1 || h > height + padding - 1)
                        {
                            extendedMatrix[d, h, w] = defaultelementGenerator();
                        }
                        else
                        {
                            extendedMatrix[d, h, w] = matrix[d, h - padding, w - padding];
                        }
                    }
                }
            }
            return extendedMatrix;
        }

        public static double[,] AddMatrix(this double[,] matrix, double[,] other)
        {
#if DEBUG
            if (matrix.GetLength(0) != other.GetLength(0)) throw new ArgumentException("Matrix Dimension mismatch!");
            if (matrix.GetLength(1) != other.GetLength(1)) throw new ArgumentException("Matrix Dimension mismatch!");
#endif
            int dimw = matrix.GetLength(1); int dimh = matrix.GetLength(0);
            double[,] rval = new double[dimh, dimw];
#if DEBUG
            for (int i = 0; i < dimh; i++)
            {
                for (int j = 0; j < dimw; j++)
                {
                    rval[i, j] = matrix[i, j] + other[i, j];
                }
            }
#else
            Parallel.For(0, dimh, (i) =>
            {
                Parallel.For(0, dimw, (j) =>
                {
                    rval[i, j] = matrix[i, j] + other[i, j];
                });
            });
#endif
            return rval;
        }

        public static double[,,] AddMatrix(this double[,,] matrix, double[,,] other)
        {
#if DEBUG
            if (matrix.GetLength(0) != other.GetLength(0)) throw new ArgumentException("Matrix Dimension mismatch!");
            if (matrix.GetLength(1) != other.GetLength(1)) throw new ArgumentException("Matrix Dimension mismatch!");
            if (matrix.GetLength(2) != other.GetLength(2)) throw new ArgumentException("Matrix Dimension mismatch!");
#endif
            int dimd = matrix.GetLength(0);
            int dimh = matrix.GetLength(1);
            int dimw = matrix.GetLength(2);
            double[,,] rval = (double[,,])matrix.Clone();
#if DEBUG
            for (int k = 0; k < dimd; k++)
            {
                for (int i = 0; i < dimh; i++)
                {
                    for (int j = 0; j < dimw; j++)
                    {
                        rval[k, i, j] = matrix[k, i, j] + other[k, i, j];
                    }
                }
            }
#else
            for (int k = 0; k < dimd; k++)
            {
                Parallel.For(0, dimh, (i) =>
                {
                    Parallel.For(0, dimw, (j) =>
                    {
                        rval[k, i, j] = matrix[k, i, j] + other[k, i, j];
                    });
                });
            }
#endif
            return rval;
        }

        public static double[,,] AddMatrix(this double[,,] matrix, double[,] other, int depth)
        {
#if DEBUG
            if (matrix.GetLength(1) != other.GetLength(0)) throw new ArgumentException("Matrix Dimension mismatch!");
            if (matrix.GetLength(2) != other.GetLength(1)) throw new ArgumentException("Matrix Dimension mismatch!");
#endif
            int dimh = matrix.GetLength(1);
            int dimw = matrix.GetLength(2);
            double[,,] rval = (double[,,])matrix.Clone();
#if DEBUG
            for (int i = 0; i < dimh; i++)
            {
                for (int j = 0; j < dimw; j++)
                {
                    rval[depth, i, j] = matrix[depth, i, j] + other[i, j];
                }
            }
#else
            Parallel.For(0, dimh, (i) =>
            {
                Parallel.For(0, dimw, (j) =>
                {
                    rval[depth, i, j] = matrix[depth, i, j] + other[i, j];
                });
            });
#endif
            return rval;
        }

        public static string StrigifyMatrix(this double[,,] m)
        {
            int d0 = m.GetLength(0);
            int d1 = m.GetLength(1);
            int d2 = m.GetLength(2);
            string rval = $"{d0}x{d1}x{d2}\n";
            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        rval += m[i, j, k] >= 0 ? " " : "";
                        rval += ($"{m[i, j, k].ToString("F3")},");
                    }
                    rval += "\n";
                }
                rval += "\n";
            }
            return rval;
        }

        public static string StrigifyMatrix(this double[,] m)
        {
            int d0 = m.GetLength(0);
            int d1 = m.GetLength(1);
            string rval = $"{d0}x{d1}\n";
            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    rval += m[i, j] >= 0 ? " " : "";
                    rval += ($"{m[i, j].ToString("F3")},");
                }
                rval += "\n";
            }
            return rval;
        }

        public static string StringyfyVector(this double[] v)
        {
            return v.Select(a => a.ToString("F3")).Aggregate((a, b) => $"{a}, {b}");
        }
    }
}
