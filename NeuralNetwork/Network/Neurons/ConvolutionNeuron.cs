using System;
using System.IO;

namespace NeuralNetwork
{
    internal class ConvolutionNeuron : Neuron
    {
        internal double DeltaO = double.NaN;
        readonly ConvolutionLayer Layer;

        public ConvolutionNeuron(ConvolutionLayer myLayer, long id) : base(NeuronType.ConvolutionNeuron, id)
        {
            Layer = myLayer;
        }

        private ConvolutionNeuron(ConvolutionNeuron prototype) : base(prototype.Type, prototype.ID)
        {
            Layer = prototype.Layer;
        }

        internal override void ResetDelta()
        {
            DeltaO = double.NaN;
            base.ResetDelta();
        }

        internal override double GetDelta(double targetvalue, Neuron caller = null)
        {
            if (caller == null) throw new ArgumentNullException();
            if (caller == this) throw new ArgumentException();
            return Layer.GetDeltaForNeuron(caller);
        }

        internal override void Backpropagate(double targetvalue)
        {
            DeltaO = 0;
            if (outEdges.Count == 0)
            {
                DeltaO = Network.dNetworkLossFunction(targetvalue, Value);
            }
            else
            {
                foreach (var edge in outEdges)
                {
                    DeltaO *= edge.Destination.GetDelta(targetvalue, this);
                }
            }
        }
    }
}
