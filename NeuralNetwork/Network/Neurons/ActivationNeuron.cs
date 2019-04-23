using System;
using System.IO;

namespace NeuralNetwork
{
    public enum ActivationFunctionType
    {
        Sigmoid,
        Tanh,
        ReLu,
    }
    internal class ActivationNeuron : Neuron
    {
        public delegate double ActivationFunction(double x);

        internal ActivationFunctionType ActivationFunType { get; set; }
        private readonly ActivationFunction fun;
        private readonly ActivationFunction dfun;
        private readonly bool scaling;

        public ActivationNeuron(Layer parent, ActivationFunctionType funtype, bool scale, long id) : base(NeuronType.ActivationNeuron, parent, id)
        {
            scaling = scale;
            ActivationFunType = funtype;
            switch (funtype)
            {
                case ActivationFunctionType.ReLu:
                    fun = ReLu;
                    dfun = dReLu;
                    break;
                case ActivationFunctionType.Sigmoid:
                    fun = Sigmoid;
                    dfun = dSigmoid;
                    break;
                case ActivationFunctionType.Tanh:
                    fun = Math.Tanh;
                    dfun = dTanh;
                    break;
            }
        }

        public override void CalculateValue()
        {
            double newvalue = 0;
            foreach (var edge in inEdges)
            {
                newvalue += edge.Origin.Value * edge.Weight;
            }

            if (scaling)
                newvalue = LinScale(newvalue);

            DerivateValue = dfun(newvalue);
            Value = fun(newvalue);
        }

        internal override void Backpropagate(double targetvalue)
        {
            foreach (var edge in inEdges)
            {
                edge.UpdateWeight(targetvalue);
            }
        }

        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        private double dSigmoid(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        private double ReLu(double x)
        {
            return Math.Max(0, x);
        }

        private double dReLu(double x)
        {
            return x > 0 ? 1 : 0;
        }

        private double dTanh(double x)
        {
            return 1 - Math.Tanh(x) * Math.Tanh(x);
        }

        private double LinScale(double x)
        {
            return x / inEdges.Count;
        }
    }
}
