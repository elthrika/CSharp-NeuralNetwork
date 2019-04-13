using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    internal class ReLuLayer : Layer
    {
        public ReLuLayer(Layer previousLayer, NeuronSource s) : base(LayerType.ReLu, s)
        {
            Size = previousLayer.Size;
            AllNeurons = new ActivationNeuron[Size];

            IReadOnlyList<Neuron> previousNeurons = previousLayer.GetNeurons();
            for (int i = 0; i < Size; i++)
            {
                AllNeurons[i] = Source.GetActivationNeuron(ActivationFunctionType.ReLu);
                Edge e = Source.MakeEdge(1, previousNeurons[i], AllNeurons[i]);
            }
        }

        internal override void EvaluateAllNeurons()
        {
            foreach (var neuron in AllNeurons)
            {
                neuron.CalculateValue();
            }
        }

        internal override void Backpropagate(double[] targetvalues = null)
        {
            
        }

        public override string ToString()
        {
            return $"ReLu Layer of Size: {Size}";
        }
    }
}
