package com.eyal.neural.visualize.classifier;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.eyal.neural.visualize.data.BenchmarkDataset;

public class SingleNodeClassifierTest extends BaseClassifierTest {

	private static final int DATA_POINTS = 200;
	private static final float POINT_WIDTH = 5.0f;
	private static final int ITERATIONS = 100;

	private final BenchmarkDataset dataset;
	
	public SingleNodeClassifierTest(BenchmarkDataset dataset) {
		this.dataset = dataset;
	}

	public void run() throws Exception {

		System.err.println("Starting...");

		try (RecordReader recordReader = new CSVRecordReader(0, ',')) {

			System.err.println("Creating Dataset...");
			DataSet allData = createDataSet(dataset, recordReader, DATA_POINTS);

			System.err.println("Creating Neural Network...");
			MultiLayerConfiguration configuration = createNetworkConfiguration();

			MultiLayerNetwork model = new MultiLayerNetwork(configuration);
			model.init();
			
			System.err.println("Training...");
			model.fit(allData);

			System.err.println("Plotting...");
			plotDataWithJzy3d(model, allData, POINT_WIDTH);

			// test the trained model with entire data-set: ---------------------
			INDArray output = model.output(allData.getFeatureMatrix());
			Evaluation eval = new Evaluation(dataset.classesCount);
			eval.eval(allData.getLabels(), output);
			System.err.println(eval);
		}
	}

	private MultiLayerConfiguration createNetworkConfiguration() {
		
		MultiLayerConfiguration configuration 
		= new NeuralNetConfiguration.Builder()

			// network/training parameters:
			.iterations(ITERATIONS)
			.activation(Activation.SOFTMAX)
			.weightInit(WeightInit.XAVIER)
			.learningRate(0.1)
			.regularization(true).l2(0.0001)
	
			.list()
			// single layer/node:
			.layer(0, new OutputLayer.Builder(
					LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
					.activation(Activation.SOFTMAX)
					.nIn(dataset.featuresCount).nOut(dataset.classesCount).build())

			// training config:
			.backprop(true).pretrain(false)
			.build();
		
		return configuration;
	}
	
	public static void main(String[] args) throws Exception {
		
		BenchmarkDataset dataset = BenchmarkDataset.IRIS_2D_DATASET;
		//BenchmarkDataset dataset = BenchmarkDataset.CLOUDS_DATASET;
		//BenchmarkDataset dataset = BenchmarkDataset.CONCENTRIC_DATASET;
		//BenchmarkDataset dataset = BenchmarkDataset.GAUSS_2D_DATASET;
		
		new SingleNodeClassifierTest(dataset).run();
	}
}
