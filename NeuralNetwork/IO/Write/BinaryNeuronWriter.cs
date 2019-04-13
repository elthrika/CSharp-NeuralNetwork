using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.IO
{
    class BinaryNeuronWriter : NeuronWriter
    {
        BinaryWriter Writer;

        internal BinaryNeuronWriter(FileStream fs)
        {
            Writer = new BinaryWriter(fs);
        }

        public void WriteNeuron(Neuron n)
        {
            switch (n.Type)
            {
                case Neuron.NeuronType.ActivationNeuron:
                    WriteNeuron(n as ActivationNeuron);
                    break;
                case Neuron.NeuronType.BiasNeuron:
                    WriteNeuron(n as BiasNeuron);
                    break;
                case Neuron.NeuronType.ConvolutionNeuron:
                    WriteNeuron(n as ConvolutionNeuron);
                    break;
                case Neuron.NeuronType.InputNeuron:
                    WriteNeuron(n as InputNeuron);
                    break;
                case Neuron.NeuronType.PoolingNeuron:
                    WriteNeuron(n as PoolingNeuron);
                    break;
                case Neuron.NeuronType.SoftMaxNeuron:
                    WriteNeuron(n as SoftMaxNeuron);
                    break;
            }
        }

        public void WriteNeuron(ActivationNeuron n)
        {
            Writer.Write((int)n.Type);
            Writer.Write(n.ID);
        }

        public void WriteNeuron(BiasNeuron n)
        {
            Writer.Write((int)n.Type);
            Writer.Write(n.ID);
            Writer.Write(n.Value);
        }

        public void WriteNeuron(ConvolutionNeuron n)
        {
            Writer.Write((int)n.Type);
            Writer.Write(n.ID);
        }

        public void WriteNeuron(InputNeuron n)
        {
            Writer.Write((int)n.Type);
            Writer.Write(n.ID);
        }

        public void WriteNeuron(PoolingNeuron n)
        {
            Writer.Write((int)n.Type);
            Writer.Write(n.ID);
        }

        public void WriteNeuron(SoftMaxNeuron n)
        {
            Writer.Write((int)n.Type);
            Writer.Write(n.ID);
        }
    }
}
