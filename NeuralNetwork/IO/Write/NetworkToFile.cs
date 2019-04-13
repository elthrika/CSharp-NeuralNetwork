using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.IO
{
    class NetworkToFile
    {
        internal enum WriteMode
        {
            Binary,
            PlainText
        }

        Network nn;

        public NetworkToFile(Network nn)
        {
            this.nn = nn;
        }

        internal void Write(WriteMode mode, string filename)
        {
            using (FileStream fs = File.OpenWrite(filename))
            {
                switch (mode)
                {
                    case WriteMode.Binary:
                        Write(new BinaryLayerWriter(fs), new BinaryNeuronWriter(fs), new BinaryEdgeWriter(fs));
                        break;
                    case WriteMode.PlainText:
                        //WritePlainText(fs);
                        break;
                }
            }
        }

        private void Write(LayerWriter lw, NeuronWriter nw, EdgeWriter ew)
        {
            lw.WriteNumber(1712);
            lw.WriteNumber((int)nn.lossFunctionType);

            var layers = nn.GetLayers();
            lw.WriteNumber(layers.Count());
            foreach (var layer in layers.Reverse())
            {
                lw.WriteLayer(layer);
            }
            var neurons = nn.Source.GetGeneratedNeurons();
            foreach (var n in neurons.Values)
            {
                nw.WriteNeuron(n);
            }
            var edges = nn.Source.GetGeneratedEdges();
            foreach (Edge e in edges)
            {
                ew.WriteEdge(e);
            }
        }
    }
}
