using System;
using System.IO;

namespace NeuralNetwork
{
    internal class SoftMaxNeuron : Neuron
    {
        internal SoftMaxNeuron(Layer parent, long id) : base(NeuronType.SoftMaxNeuron, parent, id)
        {

        }

        internal void SetDelta(double delta)
        {
            _delta = delta;
        }

        internal override double GetDelta(double targetvalue, Neuron caller = null)
        {
            return _delta;
        }
    }
}
