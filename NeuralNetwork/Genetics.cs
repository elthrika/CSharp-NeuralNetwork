using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class Genetics
    {
        public double MutationRate { get; set; }
        public double MutationDelta { get; set; }


        internal Genetics(double mr, double mDelta)
        {
            MutationRate = mr;
            MutationDelta = mDelta;
        }

        void Mutate(double[] x)
        {
            for (int i = 0; i < x.Length; i++)
            {
                if(Helper.Random < MutationRate)
                {
                    x[i] *= MutationDelta;
                }
            }
        }

        double[] Crossover(double[] x, double[] y)
        {
#if DEBUG
            System.Diagnostics.Debug.Assert(x.Length == y.Length);
#endif
            double[] crossover = new double[x.Length];
            int crossoverpoint = Helper.RandomInt(x.Length);
            for (int i = 0; i < crossoverpoint; i++)
            {
                crossover[i] = x[i];
            }
            for (int i = crossoverpoint+1; i < x.Length; i++)
            {
                crossover[i] = y[i];
            }
            return crossover;
        }

    }
}
