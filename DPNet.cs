using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

using ElemType = System.Double;
using Status = System.Int32;

namespace ConsoleApplication1
{
    class DPNet
    {

        protected int layerNum;
        protected int[] nodeNumEachLayer;
        protected double[,,] weight;
        protected double[,,] deltaWeight;
        protected ElemType[,] node;
        protected ElemType[,] delta;
        protected ElemType[] testNumber;
        protected ElemType[,] input;
        protected ElemType[,] output;
        protected int sampleNum;
        private double learningRate;
        private double learningMomentum;

        public double LearningRate {
            get
            {
                return learningRate;
            }
            set
            {
                if (learningRate <= 1.0 || learningRate > 0)
                {
                    learningRate = value;
                }
                else
                {
                    Console.WriteLine("学习速率为非法值，本次改动无效");
                }
            }
        }

        public double LearningMomentum
        {
            get
            {
                return learningMomentum;
            }
            set
            {
                if (learningRate <= 1.0 || learningRate > 0)
                {
                    learningMomentum = value;
                }
                else
                {
                    Console.WriteLine("学习动量为非法值，本次改动无效");
                }
            }
        }

        public DPNet()
        {
            layerNum = 0;
            learningRate = 0.7;
            learningMomentum = 0.9;
            sampleNum = 0;
            nodeNumEachLayer = null;
            weight = null;
            deltaWeight = null;
            delta = null;
            testNumber = null;
            node = null;
            input = null;
            output = null;
        }

        public Status CreateNet(params int[] nums)
        {
            //FreeNet
            if (nums[0] <= 2)
            {
                Console.WriteLine("非法的神经网络层数");
                return -1;
            }

            layerNum = nums[0];
            nodeNumEachLayer = new int[layerNum];
            int maxNode = 0;
            for(int i=0;i<layerNum;i++)
            {
                if (nums[i + 1] >= 1)
                {
                    if (maxNode < nums[i + 1]) maxNode = nums[i + 1];
                    nodeNumEachLayer[i] = nums[i + 1] + 1;
                }
                else
                {
                    Console.WriteLine("非法的节点数");
                    return -1;
                }
            }
            maxNode++;
            weight = new double[layerNum, maxNode, maxNode];
            deltaWeight = new double[layerNum, maxNode, maxNode];

            Random ra = new Random();

            for(int i=0;i<layerNum-1;i++)
            {
                for(int j=0;j<nodeNumEachLayer[i+1];j++)
                {
                    for(int k=0;k<nodeNumEachLayer[i];k++)
                    {
                        do
                        {
                            weight[i, j, k] = ra.NextDouble() * 2 - 1.0;
                        }
                        while (Math.Abs(weight[i, j, k]) < 1e-6);
                        deltaWeight[i, j, k] = 0.0;
                    }
                }
            }

            node = new ElemType[layerNum, maxNode];

            delta = new ElemType[layerNum, maxNode];
            
            return 0;
        }

        public Status LoadSimplesFromFile(String fileName) 
        {
            //Free
            string fileContent = File.ReadAllText(fileName);
            if(fileContent==null)
            {
                Console.WriteLine("未能成功打开样例文件");
                return -1;
            }
            string[] integerStrings = fileContent.Split
                (new char[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
          
            sampleNum = int.Parse(integerStrings[0]);
            int inputNodeNum = int.Parse(integerStrings[1]);
            int outputNodeNum = int.Parse(integerStrings[2]);

            input = new ElemType[sampleNum,inputNodeNum];
            output = new ElemType[sampleNum, outputNodeNum];

            if(input==null || output==null)
            {
                Console.WriteLine("创建input output数组失败");
                return -1;
            }
            int numIndex = 3;
            for (int i = 0; i < sampleNum; i++)
            {
                for(int j=0;j<inputNodeNum;j++)
                {
                    input[i, j] = ElemType.Parse(integerStrings[numIndex]);
                    numIndex++;
                }
                for(int k=0;k<outputNodeNum;k++)
                {
                    output[i, k] = ElemType.Parse(integerStrings[numIndex]);
                    numIndex++;
                }
            }
        
            return 0;
        }

        public double TrainSingleSample(int sampleIndex)
        {
            int i;
            for(i=0;i<nodeNumEachLayer[0]-1;i++)
            {
                node[0, i] = input[sampleIndex, i];
            }

            node[0, nodeNumEachLayer[0]-1] = 1;

            for(i=1;i<layerNum;i++)
            {
                int j;
                for (j=0;j<nodeNumEachLayer[i]-1;j++)
                {
                    node[i, j] = 0;
                    for(int k=0;k<nodeNumEachLayer[i-1];k++)
                    {
                        node[i, j] += node[i - 1, k] * weight[i - 1,j,k];
                    }
                    node[i, j] /= (nodeNumEachLayer[i - 1]);
                    node[i, j] = 1.0 / (1.0 + Math.Exp(-node[i, j]));
                }
                node[i, j] = 1;
            }

            i = layerNum - 1;
            for(int j=0;j<nodeNumEachLayer[i]-1;j++)
            {
                delta[i, j] = node[i, j] * (1 - node[i, j]) * (node[i, j] - output[sampleIndex, j]);
            }

            int avoidThreshold = 1;
            for (i = layerNum - 2; i > 0; i--)
            {
                for(int j=0;j<nodeNumEachLayer[i];j++)
                {
                    delta[i, j] = 0;
                    int k = 0;
                    for(k=0;k<nodeNumEachLayer[i+1]-avoidThreshold;k++)
                    {
                        delta[i, j] += (weight[i, k, j] * delta[i + 1, k]);//试一下
                    }
                    delta[i, j] *= (node[i, k] * (1.0 - node[i, k]));
                }
                avoidThreshold = 0;
            }

            for(i=0;i<layerNum-1;i++)
            {
                if (i == layerNum - 2)
                    avoidThreshold = 1;
                for(int j=0;j<nodeNumEachLayer[i+1]-avoidThreshold;j++)
                {
                    for(int k=0;k<nodeNumEachLayer[i];k++)
                    {
                        weight[i,j,k] += learningMomentum * deltaWeight[i, j, k];
                        weight[i, j, k] -= learningRate * delta[i + 1, j] * node[i, k];
                        deltaWeight[i, j, k] = learningMomentum * deltaWeight[i, j, k] - learningRate * delta[i + 1, j] * node[i, k];                    }
                }
            }

            ElemType error = 0;
            for(i=0;i<nodeNumEachLayer[layerNum-1]-1;i++)
            {
                error += Math.Pow((node[layerNum - 1, i] - output[sampleIndex,i]), 2.0);
            }
            error /= 2.0;
            
            return error;
        }

        public Status Train(int maxTurn, ElemType allowedError)
        {

            for(int i=0;i<maxTurn;i++)
            {
                Console.WriteLine("正在进行第"+(i + 1)+ "次训练");
                ElemType error = 0;
                for(int j=0;j<sampleNum;j++)
                {
                    error += TrainSingleSample(j);
                }
                error /= sampleNum;
                Console.WriteLine("本轮误差"+error);

                if (error < allowedError)
                {
                    Console.WriteLine("已达到允许误差值");

                    return 0;
                }
            }
            Console.WriteLine("已达到最大训练数量");
            return 0;
        }

        public Status Test(ElemType[] testInput,ElemType[] testOutput)
        {
            int i;
            for (i = 0; i < nodeNumEachLayer[0] - 1; i++)
            {
                node[0, i] = testInput[i];
            }

            node[0, nodeNumEachLayer[0] - 1] = 1;

            for (i = 1; i < layerNum; i++)
            {
                int j;
                for (j = 0; j < nodeNumEachLayer[i] - 1; j++)
                {
                    node[i, j] = 0;
                    for (int k = 0; k < nodeNumEachLayer[i - 1]; k++)
                    {
                        node[i, j] += node[i - 1, k] * weight[i - 1, j, k];
                    }
                    node[i, j] /= (nodeNumEachLayer[i - 1]);
                    node[i, j] = 1.0 / (1.0 + Math.Exp(-node[i, j]));
                }
                node[i, j] = 1;
            }
            for(i=0;i<nodeNumEachLayer[layerNum-1]-1;i++)
            {
                testOutput[i] = node[layerNum - 1, i];
            }
            return 0;
        }


    }
}
