using System;
using System.IO;

namespace NeuralNetwork
{
    internal class ConvolutionNeuron : Neuron
    {
        internal double DeltaO = double.NaN;

        public ConvolutionNeuron(Layer parent, long id) : base(NeuronType.ConvolutionNeuron, parent, id)
        {
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
            return ((ConvolutionLayer)Parent).GetDeltaForNeuron(caller);
        }

        internal override void Backpropagate(double targetvalue)
        {
            DeltaO = 0;
            if (outEdges.Count == 0)
            {
                DeltaO = Parent.Parent.dNetworkLossFunction(targetvalue, Value);
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
