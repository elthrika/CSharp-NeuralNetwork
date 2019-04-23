using System;
using System.IO;

namespace NeuralNetwork
{
    internal class InputNeuron : Neuron
    {
        internal InputNeuron(Layer parent, long id) : base(NeuronType.InputNeuron, parent, id)
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
