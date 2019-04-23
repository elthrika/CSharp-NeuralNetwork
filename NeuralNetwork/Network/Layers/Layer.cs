using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
    internal abstract class Layer
    {
        public enum LayerType
        {
            InputLayer,
            FullyConnected,
            SoftMax,
            Pooling,
            Convolutional,
            ReLu
        }

        protected Neuron[] AllNeurons;
        internal readonly LayerType Type;
        public int Size { get; protected set; }
        protected NeuronSource Source;
        internal Network Parent;

        internal abstract void EvaluateAllNeurons();
        
        protected Layer(LayerType type, Network parent)
        {
            Parent = parent;
            Source = parent.Source;
            Type = type;
        }

        public double[] Outputs()
        {
            var outputs = new double[Size];
            for (int i = 0; i < Size; i++)
            {
                outputs[i] = AllNeurons[i].Value;
            }
            return outputs;
        }

        public IReadOnlyList<Neuron> GetNeurons()
        {
            return Array.AsReadOnly(AllNeurons);
        }

        internal abstract void Backpropagate(double[] targetvalues = null);       
    }
}
