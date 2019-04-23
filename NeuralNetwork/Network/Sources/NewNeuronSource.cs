using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class NewNeuronSource : NeuronSource
    {
        long curID = 0;

        Dictionary<long, Neuron> AllGenerated = new Dictionary<long, Neuron>();
        List<Edge> Edges = new List<Edge>();
        
        public IReadOnlyDictionary<long, Neuron> GetGeneratedNeurons()
        {
            return AllGenerated;
        }

        public IReadOnlyList<Edge> GetGeneratedEdges()
        {
            return Edges.AsReadOnly();
        }

        public ActivationNeuron GetActivationNeuron(Layer parent, ActivationFunctionType funtype, bool scale = true, long id = -1)
        {
            var n = new ActivationNeuron(parent, funtype, scale, GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public BiasNeuron GetBiasNeuron(Layer parent, double constantVal = 1, long id = -1)
        {
            var n = new BiasNeuron(parent, constantVal, GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public ConvolutionNeuron GetConvolutionNeuron(Layer parent, long id = -1)
        {
            var n = new ConvolutionNeuron(parent, GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public InputNeuron GetInputNeuron(Layer parent, long id = -1)
        {
            var n = new InputNeuron(parent, GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public PoolingNeuron GetPoolingNeuron(Layer parent, long id = -1)
        {
            var n = new PoolingNeuron(parent, GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public SoftMaxNeuron GetSoftMaxNeuron(Layer parent, long id = -1)
        {
            var n = new SoftMaxNeuron(parent, GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public long GetID(long id)
        {
            curID++;
            return id > 0 ? id : curID;
        }

        public Edge MakeEdge(double weight, Neuron origin, Neuron destination)
        {
            Edge e = new Edge(weight, origin, destination);
            Edges.Add(e);
            return e;
        }

        public Edge MakeEdge(double weight, long originID, long destinationID)
        {
            var origin = AllGenerated[originID];
            var destination = AllGenerated[destinationID];
            return MakeEdge(weight, origin, destination);
        }
    }
}
