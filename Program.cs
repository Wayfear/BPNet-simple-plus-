using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ElemType = System.Double;

namespace ConsoleApplication1
{
    class Program
    {
        static void Main(string[] args)
        {
            DPNet dpNet = new DPNet();
            String filename = "C:\\Users\\xuan\\Documents\\sample.txt";
            dpNet.LoadSimplesFromFile(filename);
            dpNet.CreateNet(3, 2, 8, 1);
            dpNet.Train(9999,1e-5);
            //dpNet.LearningMomentum=0.5;
            ElemType[] input = new ElemType[2];
            ElemType[] output = new ElemType[1];
            while(true)
            {
                string s = Console.ReadLine();
                string[] sGroup = s.Split(' ');
                input[0] = double.Parse(sGroup[0]);
                input[1] = double.Parse(sGroup[1]);
                dpNet.Test(input, output);
                Console.WriteLine(output[0]);
            }
           
        }
    }
}
