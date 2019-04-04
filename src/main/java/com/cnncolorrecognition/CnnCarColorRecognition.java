/**
 * 
 */
package com.cnncolorrecognition;

import java.io.File;
import java.util.Random;

import static java.lang.Math.toIntExact;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

/**
 * @author Arnaud
 *
 */
public class CnnCarColorRecognition {

	protected static int height = 227;
    protected static int width = 227;
    protected static int channels = 3;
    protected static int batchSize = 20;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int epochs = 100;
    protected static double splitTrainTest = 0.7;
    protected static boolean save = true;
    protected static int maxPathsPerLabel = 30; 
    private int numLabels;
    
    public void execute(String[] args) throws Exception {
    	
    	System.out.println("Load data....");
    	
    	// This will return as label the base name of the parent file of the path (the directory).
    	ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    	
    	// Getting the main path
        File mainPath = new File(System.getProperty("user.dir"), "/src/main/resources/vehicle_images/");
        
        // Split up a root directory in to files
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        
        // Get the total number of pictures
        int numExamples = toIntExact(fileSplit.length());
        
        // Get the total number of labels
        // This only works if the root directory is clean, meaning it contains only label sub directories.
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; 
        
        // Randomizes the order of paths in an array and removes paths randomly to have the same number of paths for each label.
        // Further interlaces the paths on output based on their labels, to obtain easily optimal batches for training.
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathsPerLabel);
         
        // Get the list of loadable locations exposed as an iterator.
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        
        // Get the training data
        InputSplit trainData = inputSplit[0];
        
        // Get the test data
        InputSplit testData = inputSplit[1];
        
        // Interface for data normalizers. Data normalizers compute some sort of statistics over a dataset and scale the data in some way.
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        
        System.out.println("Build model....");
        
        // ComputationGraph network for the complex architecture described in the paper: "Vehicle Color Recognition using Convolutional
        // Neural Network" written by Reza Fuad Rachmadi and I Ketut Eddy Purnama. 
        ComputationGraph myNetwork =  customizedNetwork();
        
        // Initialize the ComputationGraph network
        myNetwork.init();
        
        // This will read a local file system and parse images of a given height and width.
        // All images are rescaled and converted to the given height, width, and number of channels.
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        
        // Will handle traversing through the dataset and preparing the data for a neural network.
        DataSetIterator dataIter;
        
        System.out.println("Train model....");
        
        // Train without initial transformations
        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        long startTime = System.currentTimeMillis();
        myNetwork.fit(dataIter,epochs);
        System.out.println("Training time...: " + (System.currentTimeMillis() - startTime) / 1000.0 + "s");
        
        System.out.println("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = myNetwork.evaluate(dataIter);
        System.out.println(eval.stats(true));
        
        System.out.println("****************End of Training********************");
        
        // Saving the model in a .bin file
        if (save) {
        	System.out.print("Saving model....");
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "/github/Car-color-recognition-using-CNN-master/src/main/resources/");
            ModelSerializer.writeModel(myNetwork, basePath + "model.bin", true);
        }       
    }
    
    // First convolutional operation
    private ConvolutionLayer convInit(int in, // number of channels
            int out, // number of kernels or filters
            int[] kernel, // width and height of the kernel
            int[] stride, // width and height of the stride
            int[] pad,    //width and height of the pad
            double bias) {

            return new ConvolutionLayer.Builder(kernel, stride, pad) 
                     .nIn(in)
                     .nOut(out)
                     .biasInit(bias)
                     .build();
    }
    
    // Main convolutional operation
    private ConvolutionLayer conv(int out, // number of kernels or filters
            double bias) {
    	
    	return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1})
        		.nOut(out)
        		.biasInit(bias)
        		.build();
    }
    
    // Pooling / subsampling
    private SubsamplingLayer maxPool(int[] kernel) {
    	return new SubsamplingLayer.Builder(kernel, 
                new int[]{2,2}).build();
    	          //stride of 2
    }
    
    // Fully connected layer
    private DenseLayer fullyConnected(int out, 
            double bias, 
            double dropOut, 
            Distribution dist) {
    	
    	return new DenseLayer.Builder()
        		.nOut(out)
        		.biasInit(bias)
        		.dropOut(dropOut)
        		.dist(dist)
        		.build();
    }
    
    
    public ComputationGraph customizedNetwork() {
    	double nonZeroBias = 1;
        double dropOut = 0.5; 
        
        // Configuring the network
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
        		.seed(seed)
        		.weightInit(WeightInit.DISTRIBUTION)
        		.dist(new NormalDistribution(0.0, 0.01))
        		.activation(Activation.RELU)
        		.updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
        		.biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
        		.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
        		.l2(5 * 1e-4)
        		.graphBuilder()
        		.addInputs("input")
        		.setInputTypes(InputType.convolutional(height, width, channels))
        		.addLayer("top_cnn1", convInit(channels, 48, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0), "input")
        		.addLayer("top_lrn1", new LocalResponseNormalization.Builder().build(), "top_cnn1")       		
        		.addLayer("top_maxpool1", maxPool(new int[]{3,3}), "top_lrn1")       		
        		.addLayer("top_top_cnn2", conv(64, nonZeroBias), "top_maxpool1")
        		.addLayer("top_top_lrn2", new LocalResponseNormalization.Builder().build(), "top_top_cnn2")
        		.addLayer("top_top_maxpool2", maxPool(new int[]{3,3}), "top_top_lrn2")
        		.addLayer("top_bottom_cnn2", conv(64, nonZeroBias), "top_maxpool1")
        		.addLayer("top_bottom_lrn2", new LocalResponseNormalization.Builder().build(), "top_bottom_cnn2")
        		.addLayer("top_bottom_maxpool2", maxPool(new int[]{3,3}), "top_bottom_lrn2")       		
        		.addVertex("top_merge", new MergeVertex(), "top_top_maxpool2", "top_bottom_maxpool2")        		
        		.addLayer("top_cnn3", conv(192, 0), "top_merge")         		
        		.addLayer("top_top_cnn4", conv(96, nonZeroBias), "top_cnn3") 		
        		.addLayer("top_top_cnn5", conv(64, nonZeroBias), "top_top_cnn4") 
        		.addLayer("top_top_maxpool3", maxPool(new int[]{3,3}), "top_top_cnn5")
        		.addLayer("top_bottom_cnn4", conv(96, nonZeroBias), "top_cnn3")
        		.addLayer("top_bottom_cnn5", conv(64, nonZeroBias), "top_bottom_cnn4")
        		.addLayer("top_bottom_maxpool3", maxPool(new int[]{3,3}), "top_bottom_cnn5") 
        		
        		.addLayer("bottom_cnn1", convInit(channels, 48, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0), "input")
        		.addLayer("bottom_lrn1", new LocalResponseNormalization.Builder().build(), "bottom_cnn1")        		
        		.addLayer("bottom_maxpool1", maxPool(new int[]{3,3}), "bottom_lrn1")        		
        		.addLayer("bottom_top_cnn2", conv(64, nonZeroBias), "bottom_maxpool1")
        		.addLayer("bottom_top_lrn2", new LocalResponseNormalization.Builder().build(), "bottom_top_cnn2")
        		.addLayer("bottom_top_maxpool2", maxPool(new int[]{3,3}), "bottom_top_lrn2")
        		.addLayer("bottom_bottom_cnn2", conv(64, nonZeroBias), "bottom_maxpool1")
        		.addLayer("bottom_bottom_lrn2", new LocalResponseNormalization.Builder().build(), "bottom_bottom_cnn2")
        		.addLayer("bottom_bottom_maxpool2", maxPool(new int[]{3,3}), "bottom_bottom_lrn2")        		
        		.addVertex("bottom_merge", new MergeVertex(), "bottom_top_maxpool2", "bottom_bottom_maxpool2")        		
        		.addLayer("bottom_cnn3", conv(192, 0), "bottom_merge")       		
        		.addLayer("bottom_top_cnn4", conv(96, nonZeroBias), "bottom_cnn3")        		
        		.addLayer("bottom_top_cnn5", conv(64, nonZeroBias), "bottom_top_cnn4")
        		.addLayer("bottom_top_maxpool3", maxPool(new int[]{3,3}), "bottom_top_cnn5")
        		.addLayer("bottom_bottom_cnn4", conv(96, nonZeroBias), "bottom_cnn3")
        		.addLayer("bottom_bottom_cnn5", conv(64, nonZeroBias), "bottom_bottom_cnn4")
        		.addLayer("bottom_bottom_maxpool3", maxPool(new int[]{3,3}), "bottom_bottom_cnn5") 
        		
        		.addVertex("final_merge", new MergeVertex(), "top_top_maxpool3", "top_bottom_maxpool3", "bottom_top_maxpool3", "bottom_bottom_maxpool3")
        		.addLayer("ffn1", fullyConnected(4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)), "final_merge")
        		.addLayer("ffn2", fullyConnected(4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)), "ffn1")
        		.addLayer("final_layer", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        				.name("output")
        				.nOut(numLabels)
        				.activation(Activation.SOFTMAX)
        				.build(), "ffn2")
        		.setOutputs("final_layer")
        		.backprop(true)
        		.pretrain(false)
        		.build();
    	
    	return new ComputationGraph(conf);
    }
    
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		new CnnCarColorRecognition().execute(args);
	}
}
