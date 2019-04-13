using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;


namespace NeuralNetwork
{
    internal class SoftMaxLayer : Layer
    {
        private double[,] jacobian;

        public SoftMaxLayer(Layer previousLayer, NeuronSource s) : base(LayerType.SoftMax, s)
        {
            Size = previousLayer.Size;
            AllNeurons = new SoftMaxNeuron[Size];

            IReadOnlyList<Neuron> previousNeurons = previousLayer.GetNeurons();
            for (int i = 0; i < Size; i++)
            {
                AllNeurons[i] = Source.GetSoftMaxNeuron();
                Edge e = Source.MakeEdge(1, previousNeurons[i], AllNeurons[i]);
            }
        }

        internal override void EvaluateAllNeurons()
        {
            double[] x = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                Neuron prev = AllNeurons[i].GetInEdges()[0].Origin;
                x[i] = prev.Value;
            }
            double max = x.Max();
            double[] exps = x.Select(a => Math.Exp(a - max)).ToArray();
            double sum = exps.Sum();
            double[] softmax = exps.Select(a => a / sum).ToArray();

            jacobian = new double[Size, Size];
            for (int i = 0; i < Size; i++)
            {
                for (int j = 0; j < Size; j++)
                {
                    if(i == j)
                    {
                        jacobian[i, j] = softmax[i] * (1 - softmax[i]);
                    }
                    else
                    {
                        jacobian[i, j] = -softmax[i] * softmax[j];
                    }
                }
            }
            for (int i = 0; i < Size; i++)
            {
                AllNeurons[i].SetValue(softmax[i]);
            }            
        }

        internal override void Backpropagate(double[] targetvalues = null)
        {
            if (targetvalues == null)
            {
#if DEBUG
                for (int i = 0; i < AllNeurons.Length; i++)
                {
                    SoftMaxNeuron smn = (SoftMaxNeuron)AllNeurons[i];
                    double delta = 0;
                    for (int c = 0; c < Size; c++)
                    {
                        delta += jacobian[i, c] * Network.dNetworkLossFunction(smn.GetOutEdges()[0].Destination.GetDelta(0, smn), AllNeurons[c].Value);
                    }
                    smn.SetDelta(delta);
                }
#else
                Parallel.For(0, AllNeurons.Length, i =>
                {
                    SoftMaxNeuron smn = (SoftMaxNeuron)AllNeurons[i];
                    double delta = 0;
                    for (int c = 0; c < Size; c++)
                    {
                        delta += jacobian[i, c] * Network.dNetworkLossFunction(smn.GetOutEdges()[0].Destination.GetDelta(0, smn), AllNeurons[c].Value);
                    }
                    smn.SetDelta(delta);
                });
#endif
            }
            else
            {
#if DEBUG
                for (int i = 0; i < AllNeurons.Length; i++)
                {
                    SoftMaxNeuron smn = (SoftMaxNeuron)AllNeurons[i];
                    double delta = 0;
                    for (int c = 0; c < Size; c++)
                    {
                        delta += jacobian[i, c] * Network.dNetworkLossFunction(targetvalues[c], AllNeurons[c].Value);
                    }
                    //Console.WriteLine($"Setting delta to: {delta}");
                    smn.SetDelta(delta);

                }
#else
                Parallel.For(0, AllNeurons.Length, i =>
                {
                    SoftMaxNeuron smn = (SoftMaxNeuron)AllNeurons[i];
                    double delta = 0;
                    for (int c = 0; c < Size; c++)
                    {
                        delta += jacobian[i, c] * Network.dNetworkLossFunction(targetvalues[c], AllNeurons[c].Value);
                    }
                    //Console.WriteLine($"Setting delta to: {delta}");
                    smn.SetDelta(delta);
                });
#endif
            }
        }

        public override string ToString()
        {
            return $"Softmax Layer of Size: {Size}";
        }
    }
}
