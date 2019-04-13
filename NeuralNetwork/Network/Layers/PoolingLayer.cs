using System;
using System.IO;

namespace NeuralNetwork
{
    internal class PoolingLayer : Layer
    {
        public readonly int Tessellation;
        public (int width, int height, int depth) Dims { get; private set; }
        public (int width, int height, int depth) InputDims { get; private set; }

        public PoolingLayer(Layer previousLayer, int tessellation, NeuronSource s) : base(LayerType.Pooling, s)
        {
            Tessellation = tessellation;
            var sqrt = (int)Math.Sqrt(previousLayer.GetNeurons().Count);
            Init(previousLayer, (sqrt, sqrt, 1));
        }

        public PoolingLayer(Layer previousLayer, int tessellation, (int width, int height) dims, NeuronSource s) : base(LayerType.Pooling, s)
        {
            Tessellation = tessellation;
            Init(previousLayer, (dims.width, dims.height, 1));
        }

        public PoolingLayer(Layer previousLayer, int tessellation, (int width, int height, int depth) dims, NeuronSource s) : base(LayerType.Pooling, s)
        {
            Tessellation = tessellation;
            Init(previousLayer, dims);
        }

        private void Init(Layer previousLayer, (int width, int height, int depth) dims)
        {
            var prevNeurons = previousLayer.GetNeurons();

            if (dims.width * dims.height * dims.depth != prevNeurons.Count)
            {
                Console.WriteLine("Either layer without specified size is not square, or dimensions do not agree with the previous layer");
            }

            if (dims.width % Tessellation != 0 || dims.width % Tessellation != 0)
            {
                Console.WriteLine($"Tessellating Layer of Size: {dims} by {Tessellation}, which does not divide evenly");
            }

            int stride = dims.width;
            int xydim = dims.width * dims.height;
            int clustersPerXY = xydim / (Tessellation * Tessellation);
            Size = dims.depth * clustersPerXY;
            InputDims = dims;
            Dims = (dims.width / Tessellation, dims.height / Tessellation, dims.depth);
            AllNeurons = new PoolingNeuron[Size];

            for (int i = 0; i < Size; i++)
            {
                AllNeurons[i] = Source.GetPoolingNeuron();
            }

            for (int d = 0; d < dims.depth; d++)
            {
                for (int i = 0; i < clustersPerXY; i++)
                {
                    int st = stride * Tessellation;
                    int sdt = stride / Tessellation;
                    int clusterStartIdx = (i / sdt) * st + (i % sdt) * Tessellation;
                    clusterStartIdx += d * xydim;
                    //Console.WriteLine($"{i}: {i / sdt} * {st} + {i % sdt} * {Tessellation} = {clusterStartIdx}");
                    var inneuron = AllNeurons[i + d * clustersPerXY];
                    for (int r = 0; r < Tessellation; r++)
                    {
                        for (int c = 0; c < Tessellation; c++)
                        {
                            Neuron outneuron;
                            if (clusterStartIdx + r * stride + c < 0 || clusterStartIdx + r * stride + c >= prevNeurons.Count)
                            {
                                outneuron = Source.GetBiasNeuron(double.NegativeInfinity);
                            }
                            else
                            {
                                outneuron = prevNeurons[clusterStartIdx + r * stride + c];
                            }
                            var e = Source.MakeEdge(1, outneuron, inneuron);
                        }
                    }
                }
            }
        }

        public (int w, int h, int d) GetOutputSpatialExtent()
        {
            return Dims;
        }

        internal override void EvaluateAllNeurons()
        {
            foreach (var neuron in AllNeurons)
            {
                var inedges = neuron.GetInEdges();
                double max = double.MinValue;
                int maxidx = -1;
                for (int i = 0; i < inedges.Count; i++)
                {
                    var outn = inedges[i].Origin;
                    if(outn.Value >= max || maxidx < 0)
                    {
                        max = outn.Value;
                        maxidx = i;
                    }
                }
                neuron.SetValue(max);
                ((PoolingNeuron)neuron).MaxIdx = maxidx;
            }
        }

        internal override void Backpropagate(double[] targetvalues = null)
        {
            for (int i = 0; i < Size; i++)
            {
                double targetvalue = targetvalues == null ? 0.0D : targetvalues[i];
                AllNeurons[i].Backpropagate(targetvalue);
            }
        }

        public override string ToString()
        {
            return $"Pooling Layer with Tessellation: {Tessellation}\n{InputDims} => {Dims}";
        }
    }
}
