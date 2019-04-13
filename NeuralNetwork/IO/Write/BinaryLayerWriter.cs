using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.IO
{
    class BinaryLayerWriter : LayerWriter
    {
        BinaryWriter Writer;

        internal BinaryLayerWriter(FileStream fs)
        {
            Writer = new BinaryWriter(fs);
        }

        public void WriteLayer(Layer l)
        {
            switch (l.Type)
            {
                case Layer.LayerType.Convolutional:
                    WriteLayer(l as ConvolutionLayer);
                    break;
                case Layer.LayerType.FullyConnected:
                    WriteLayer(l as FullyConnectedLayer);
                    break;
                case Layer.LayerType.InputLayer:
                    WriteLayer(l as InputLayer);
                    break;
                case Layer.LayerType.Pooling:
                    WriteLayer(l as PoolingLayer);
                    break;
                case Layer.LayerType.ReLu:
                    WriteLayer(l as ReLuLayer);
                    break;
                case Layer.LayerType.SoftMax:
                    WriteLayer(l as SoftMaxLayer);
                    break;
            }
        }

        public void WriteLayer(ConvolutionLayer l)
        {
            Writer.Write((int)l.Type);
            Writer.Write(l.Size);
            Writer.Write(l.N_filters);
            Writer.Write(l.Filtersize);
            Writer.Write(l.Stride);
            Writer.Write(l.Padding);
            Writer.Write(l.InputWidth);
            Writer.Write(l.InputHeight);
            Writer.Write(l.InputDepth);
            for (int i = 0; i < l.filters.Count; i++)
            {
                for (int d = 0; d < l.InputDepth; d++)
                {
                    for (int h = 0; h < l.Filtersize; h++)
                    {
                        for (int w = 0; w < l.Filtersize; w++)
                        {
                            Writer.Write(l.filters[i][d, h, w]);
                        }
                    }
                }
            }
            for (int i = 0; i < l.N_filters; i++)
            {
                Writer.Write(l.biases[i]);
            }
        }

        public void WriteLayer(FullyConnectedLayer l)
        {
            Writer.Write((int)l.Type);
            Writer.Write(l.Size);
            Writer.Write(l.Scaling);
            Writer.Write((int)l.GetActivationFunctionType());

        }

        public void WriteLayer(InputLayer l)
        {
            Writer.Write((int)l.Type);
            Writer.Write(l.Size);
        }

        public void WriteLayer(PoolingLayer l)
        {
            Writer.Write((int)l.Type);
            Writer.Write(l.Size);
            Writer.Write(l.Tessellation);
            Writer.Write(l.InputDims.width);
            Writer.Write(l.InputDims.height);
            Writer.Write(l.InputDims.depth);
        }

        public void WriteLayer(ReLuLayer l)
        {
            Writer.Write((int)l.Type);
            Writer.Write(l.Size);
        }

        public void WriteLayer(SoftMaxLayer l)
        {
            Writer.Write((int)l.Type);
            Writer.Write(l.Size);
        }

        public void WriteNumber(int n)
        {
            Writer.Write(n);
        }
    }
}
