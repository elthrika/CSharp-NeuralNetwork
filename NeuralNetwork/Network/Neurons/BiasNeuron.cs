using System;
using System.IO;

namespace NeuralNetwork
{
    internal class BiasNeuron : Neuron
    {

        public BiasNeuron(double constantVal, long id) : base(NeuronType.BiasNeuron, id)
        {
            Value = constantVal;
        }

        //private BiasNeuron(BiasNeuron prototype) : base(prototype.Type, prototype.ID)
        //{
        //    Value = prototype.Value;
        //}

        public override void SetValue(double val)
        {
        }

        internal override double GetDelta(double targetvalue, Neuron caller = null)
        {
            return 0;
        }

        public override void CalculateValue()
        {
            // do nothing
        }

        internal override void Backpropagate(double targetvalue)
        {
            // do nothing, because we have no in-edges
        }

    }
}
