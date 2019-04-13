using System.IO;

namespace NeuralNetwork.IO
{
    internal class BinaryEdgeWriter : EdgeWriter
    {
        private BinaryWriter Writer;

        public BinaryEdgeWriter(FileStream fs)
        {
            Writer = new BinaryWriter(fs);
        }

        public void WriteEdge(Edge e)
        {
            Writer.Write(e.Origin.ID);
            Writer.Write(e.Destination.ID);
            Writer.Write(e.Weight);
        }
    }
}