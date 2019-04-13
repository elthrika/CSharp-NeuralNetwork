using System;
using System.IO;

namespace NeuralNetwork
{
    internal class SoftMaxNeuron : Neuron
    {
        internal SoftMaxNeuron(long id) : base(NeuronType.SoftMaxNeuron, id)
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

        internal static Neuron ReadFromFile(BinaryReader br)
        {
            NeuronType type = (NeuronType)br.ReadInt32();
            if (type != NeuronType.SoftMaxNeuron)
                throw new Exception($"SoftMaxNeuron::ReadFromFile {type} != SoftMaxNeuron");
            
            long ID = br.ReadInt64();
            var n = new SoftMaxNeuron(ID);
            return n;
        }
    }
}
