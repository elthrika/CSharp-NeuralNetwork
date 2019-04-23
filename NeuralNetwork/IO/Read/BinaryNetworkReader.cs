using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.IO
{
    class BinaryNetworkReader : NetworkReader
    {
        BinaryReader br;
        List<LayerPrototype> protos;

        public BinaryNetworkReader(BinaryReader br)
        {
            this.br = br;
            protos = new List<LayerPrototype>();
        }

        public Network Read()
        {

            var mn = br.ReadInt32();
            Console.WriteLine(mn);

            Network nn = new Network((Network.LossFunctionType)br.ReadInt32());
            nn.SetSource(new BinaryFileNeuronSource(br));

            int n_layers = br.ReadInt32();
            for (int i = 0; i < n_layers; i++)
            {
                ConstructFromFile();
            }
            foreach (var prototype in protos)
            {
                nn.Add(prototype.ToLayer(nn, nn.GetLayers().FirstOrDefault()));
            }

            while(br.BaseStream.Position < br.BaseStream.Length)
            {
                nn.Source.MakeEdge(double.NaN, -1, -1);
            }
            
            return nn;
        }

        private void ConstructFromFile()
        {
            Layer.LayerType type = (Layer.LayerType)br.ReadInt32();
            switch (type)
            {
                case Layer.LayerType.FullyConnected:
                    CreateFullyConnectedLayerPrototype();
                    return;
                case Layer.LayerType.InputLayer:
                    CreateInputLayerPrototype();
                    return;
                case Layer.LayerType.SoftMax:
                    CreateSoftMaxLayerPrototype();
                    return;
                case Layer.LayerType.Pooling:
                    CreatePoolingLayerPrototype();
                    return;
                case Layer.LayerType.Convolutional:
                    CreateConvolutionalLayerPrototype();
                    return;
                case Layer.LayerType.ReLu:
                    CreateReLuLayerPrototype();
                    return;
                default:
                    throw new ArgumentException($"Unrecognized Layer-Type: {type}");
            }
        }

        private void CreateReLuLayerPrototype()
        {
            int Size = br.ReadInt32();
            protos.Add(new LayerPrototype(Layer.LayerType.ReLu,
                ("Size", Size)
            ));
        }

        private void CreateConvolutionalLayerPrototype()
        {
            int Size = br.ReadInt32();
            int N_filters = br.ReadInt32();
            int Filtersize = br.ReadInt32();
            int Stride = br.ReadInt32();
            int Padding = br.ReadInt32();
            int InputWidth = br.ReadInt32();
            int InputHeight = br.ReadInt32();
            int InputDepth = br.ReadInt32();

            var filters = new List<double[,,]>(N_filters);
            var biases = new double[N_filters];

            for (int i = 0; i < N_filters; i++)
            {
                filters.Add(new double[InputDepth, Filtersize, Filtersize]);
                for (int d = 0; d < InputDepth; d++)
                {
                    for (int h = 0; h < Filtersize; h++)
                    {
                        for (int w = 0; w < Filtersize; w++)
                        {
                            filters[i][d, h, w] = br.ReadDouble();
                        }
                    }
                }
            }

            for (int i = 0; i < N_filters; i++)
            {
                biases[i] = br.ReadDouble();
            }

            protos.Add(new LayerPrototype(Layer.LayerType.Convolutional, 
                ("Size", Size), ("N_filters", N_filters), ("Filtersize", Filtersize), ("Stride", Stride), ("Padding", Padding), ("InputWidth", InputWidth), ("InputHeight", InputHeight), 
                ("InputDepth", InputDepth), ("filters", filters), ("biases", biases)
            ));
        }

        private void CreatePoolingLayerPrototype()
        {
            int Size = br.ReadInt32();
            int Tessellation = br.ReadInt32();
            int InputWidth = br.ReadInt32();
            int InputHeight = br.ReadInt32();
            int InputDepth = br.ReadInt32();

            protos.Add(new LayerPrototype(Layer.LayerType.Pooling,
                ("Size", Size), ("Tessellation", Tessellation), ("InputWidth", InputWidth), ("InputHeight", InputHeight), ("InputDepth", InputDepth)
            ));
        }

        private void CreateSoftMaxLayerPrototype()
        {
            int Size = br.ReadInt32();
            protos.Add(new LayerPrototype(Layer.LayerType.SoftMax,
                ("Size", Size)
            ));
        }

        private void CreateInputLayerPrototype()
        {
            int Size = br.ReadInt32();
            protos.Add(new LayerPrototype(Layer.LayerType.InputLayer,
                ("Size", Size)
            ));
        }

        private void CreateFullyConnectedLayerPrototype()
        {
            int Size = br.ReadInt32();
            bool Scaling = br.ReadBoolean();
            var funtype = (ActivationFunctionType)br.ReadInt32();

            protos.Add(new LayerPrototype(Layer.LayerType.FullyConnected, 
                ("Size", Size), ("Scaling", Scaling), ("funtype", funtype)
            ));
        }

        private struct LayerPrototype
        {
            public Dictionary<string, object> fields;
            public Layer.LayerType type;

            public LayerPrototype(Layer.LayerType t, params (string, object)[] fs)
            {
                type = t;
                fields = new Dictionary<string, object>();
                foreach (var (s, o) in fs)
                {
                    fields.Add(s, o);
                }
            }

            public T GetField<T>(string name)
            {
                return (T)fields[name];
            }

            public Layer ToLayer(Network nn, Layer previousLayer)
            {
                switch (type)
                {
                    case Layer.LayerType.FullyConnected:
                        return new FullyConnectedLayer(GetField<int>("Size"), previousLayer, nn, GetField<ActivationFunctionType>("funtype"), GetField<bool>("Scaling"));
                    case Layer.LayerType.InputLayer:
                        return new InputLayer(GetField<int>("Size"), nn);
                    case Layer.LayerType.SoftMax:
                        return new SoftMaxLayer(previousLayer, nn);
                    case Layer.LayerType.Pooling:
                        return new PoolingLayer(previousLayer, GetField<int>("Tessellation"), (GetField<int>("InputWidth"), GetField<int>("InputHeight"), GetField<int>("InputDepth")), nn);
                    case Layer.LayerType.Convolutional:
                        return new ConvolutionLayer(previousLayer, 
                            GetField<int>("N_filters"), GetField<int>("Filtersize"), GetField<int>("Stride"), GetField<int>("Padding"), 
                            (GetField<int>("InputWidth"), GetField<int>("InputHeight"), GetField<int>("InputDepth")), nn);
                    case Layer.LayerType.ReLu:
                        return new ReLuLayer(previousLayer, nn);
                    default:
                        throw new ArgumentException($"Unrecognized Layer-Type: {type}");
                }
            }
        }
    }
}
