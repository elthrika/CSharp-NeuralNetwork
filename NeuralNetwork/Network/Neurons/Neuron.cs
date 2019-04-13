using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
    internal abstract class Neuron
    {

        internal enum NeuronType
        {
            //INVALID,
            ActivationNeuron,
            BiasNeuron,
            ConvolutionNeuron,
            InputNeuron,
            PoolingNeuron,
            SoftMaxNeuron
        }


        public double Value { get; protected set; }
        public double DerivateValue { get; protected set; }

        public readonly long ID;
        internal NeuronType Type;

        internal Neuron(NeuronType type, long id)
        {
            Type = type;
            ID = id;
        }

        #region Delta
        protected double _delta = double.NaN;

        internal virtual double GetDelta(double targetvalue, Neuron caller = null)
        {
            if (!double.IsNaN(_delta)) return _delta;

            _delta = 0;

            if(outEdges.Count == 0)
            {
                //output layer
                _delta = Network.dNetworkLossFunction(targetvalue, Value) * DerivateValue;
            }
            else
            {
                //inner layer
                foreach (var edge in outEdges)
                {
                    _delta += edge.Weight * edge.Destination.GetDelta(targetvalue, this);
                }
                _delta *= DerivateValue;
            }

            return _delta;
        }

        internal virtual void ResetDelta()
        {
            _delta = double.NaN;
        }
        #endregion

        protected List<Edge> outEdges = new List<Edge>();
        protected List<Edge> inEdges = new List<Edge>();

        public virtual void CalculateValue() { }

        public virtual void SetValue(double val)
        {
            Value = val;
        }

        public virtual void SetDerivativeValue(double val)
        {
            DerivateValue = val;
        }

        public void AddInEdge(Edge e)
        {
            inEdges.Add(e);
        }

        public void AddOutEdge(Edge e)
        {
            outEdges.Add(e);
        }

        public IReadOnlyList<Edge> GetOutEdges()
        {
            return outEdges.AsReadOnly();
        }

        public IReadOnlyList<Edge> GetInEdges()
        {
            return inEdges.AsReadOnly();
        }

        internal virtual void Backpropagate(double targetvalue) { }

        internal virtual void WriteToFileBinary(BinaryWriter bw)
        {
        }

        public override string ToString()
        {
            string s = $"{Type} #{ID} - In: {inEdges.Count} Out: {outEdges.Count}";
            return s;
        }

        internal virtual void WriteToFilePlainText(StreamWriter sw)
        {
            sw.WriteLine(Type.ToString());
            sw.WriteLine(ID);
            sw.WriteLine(inEdges.Count);
            foreach (var e in inEdges)
            {
                sw.WriteLine(e.Origin.ID);
                sw.WriteLine(e.Weight);
            }

        }


        //internal static Neuron ReadFromFile(System.IO.BinaryReader br, Layer layer)
        //{
        //    NeuronType type = (NeuronType)br.ReadInt32();
        //    switch (type)
        //    {
        //        case NeuronType.ActivationNeuron:
        //            return ActivationNeuron.ReadFromFile(br, layer);
        //        case NeuronType.BiasNeuron:
        //            return BiasNeuron.ReadFromFile(br, layer);
        //        case NeuronType.ConvolutionNeuron:
        //            return ConvolutionNeuron.ReadFromFile(br, layer);
        //        case NeuronType.InputNeuron:
        //            return InputNeuron.ReadFromFile(br, layer);
        //        case NeuronType.PoolingNeuron:
        //            return PoolingNeuron.ReadFromFile(br, layer);
        //        case NeuronType.SoftMaxNeuron:
        //            return SoftMaxNeuron.ReadFromFile(br, layer);
        //        default:
        //            throw new ArgumentException($"Unknown Neuron-Type: {type}");
        //    }
        //}
    }
}
