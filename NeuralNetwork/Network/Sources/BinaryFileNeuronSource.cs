using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace NeuralNetwork
{
    class BinaryFileNeuronSource : NeuronSource
    {
        private BinaryReader Reader;
        private long curID = 0;
        private Dictionary<long, Neuron> AllGenerated;
        List<Edge> Edges;


        public BinaryFileNeuronSource(BinaryReader br)
        {
            Reader = br;
            AllGenerated = new Dictionary<long, Neuron>();
            Edges = new List<Edge>();
        }

        public ActivationNeuron GetActivationNeuron(ActivationFunctionType funtype, bool scale = true, long id = -1)
        {
            Neuron.NeuronType type = (Neuron.NeuronType)Reader.ReadInt32();
            if (type != Neuron.NeuronType.ActivationNeuron)
                throw new Exception($"ActivationNeuron::ReadFromFile {type} != ActivationNeuron");
            long ID = Reader.ReadInt64();
            var n = new ActivationNeuron(funtype, scale, GetID(ID));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public BiasNeuron GetBiasNeuron(double constantVal = 1, long id = -1)
        {
            Neuron.NeuronType type = (Neuron.NeuronType)Reader.ReadInt32();
            if (type != Neuron.NeuronType.BiasNeuron)
                throw new Exception($"BiasNeuron::ReadFromFile {type} != BiasNeuron");
            long ID = Reader.ReadInt64();
            double val = Reader.ReadDouble();
            var n = new BiasNeuron(val, ID);
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public ConvolutionNeuron GetConvolutionNeuron(ConvolutionLayer myLayer, long id = -1)
        {
            Neuron.NeuronType type = (Neuron.NeuronType)Reader.ReadInt32();
            if (type != Neuron.NeuronType.ConvolutionNeuron)
                throw new Exception($"ConvolutionNeuron::ReadFromFile {type} != ConvolutionNeuron");
            long ID = Reader.ReadInt64();
            var n = new ConvolutionNeuron(myLayer, GetID(ID));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public IReadOnlyDictionary<long, Neuron> GetGeneratedNeurons()
        {
            return AllGenerated;
        }

        public IReadOnlyList<Edge> GetGeneratedEdges()
        {
            return Edges.AsReadOnly();
        }

        public long GetID(long id)
        {
            curID++;
            return id > 0 ? id : curID;
        }

        public InputNeuron GetInputNeuron(long id = -1)
        {
            Neuron.NeuronType type = (Neuron.NeuronType)Reader.ReadInt32();
            if (type != Neuron.NeuronType.InputNeuron)
                throw new Exception($"InputNeuron::ReadFromFile {type} != InputNeuron");
            long ID = Reader.ReadInt64();
            var n = new InputNeuron(GetID(ID));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public PoolingNeuron GetPoolingNeuron(long id = -1)
        {
            Neuron.NeuronType type = (Neuron.NeuronType)Reader.ReadInt32();
            if (type != Neuron.NeuronType.PoolingNeuron)
                throw new Exception($"PoolingNeuron::ReadFromFile {type} != PoolingNeuron");
            long ID = Reader.ReadInt64();
            var n = new PoolingNeuron(GetID(ID));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public SoftMaxNeuron GetSoftMaxNeuron(long id = -1)
        {
            Neuron.NeuronType type = (Neuron.NeuronType)Reader.ReadInt32();
            if (type != Neuron.NeuronType.SoftMaxNeuron)
                throw new Exception($"SoftMaxNeuron::ReadFromFile {type} != SoftMaxNeuron");
            long ID = Reader.ReadInt64();
            var n = new SoftMaxNeuron(GetID(ID));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public Edge MakeEdge(double weight, Neuron origin, Neuron destination)
        {
            return null;
        }

        public Edge MakeEdge(double weight, long originID, long destinationID)
        {
            var origin = AllGenerated[Reader.ReadInt64()];
            var destination = AllGenerated[Reader.ReadInt64()];
            var e = new Edge(Reader.ReadDouble(), origin, destination);
            Edges.Add(e);
            return e;
        }
    }
}
