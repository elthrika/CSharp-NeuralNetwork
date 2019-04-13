using System;
using System.IO;

namespace NeuralNetwork
{
    internal class InputNeuron : Neuron
    {
        internal InputNeuron(long id) : base(NeuronType.InputNeuron, id)
        {

        }

        private InputNeuron(InputNeuron prototype) : base(prototype.Type, prototype.ID)
        {

        }

        public override void CalculateValue()
        {
            // do nothing
        }

        public override void SetValue(double val)
        {
            Value = val;
        }

        internal override void Backpropagate(double targetvalue)
        {
            // do nothing, this should never be called
            Console.Error.WriteLine("Trying to backprop on the inputlayer, this shouldn't happen");
        }
    }
}
