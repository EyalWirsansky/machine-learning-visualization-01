package com.eyal.neural.visualize.classifier;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jzy3d.colors.Color;
import org.jzy3d.maths.Coord3d;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Mapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import com.eyal.neural.visualize.data.BenchmarkDataset;
import com.eyal.neural.visualize.plot.CombinedPlotter;

public class ClassifierTest {

	private static final int SEED = 42;

	protected DataSet createDataSet(BenchmarkDataset dataset,
			RecordReader recordReader, int maxDataPoints) throws FileNotFoundException, IOException, InterruptedException {
		// read data-set from file: -------------------------------------------
		recordReader.initialize(new FileSplit(
				new ClassPathResource(dataset.fileName).getFile()));

		DataSetIterator iterator = new RecordReaderDataSetIterator(
				recordReader, dataset.fileLines, dataset.featuresCount, dataset.classesCount);

		//DataSet allData = iterator.next();
		DataSet allData = iterator.next().batchBy(maxDataPoints).get(0);

		allData.shuffle(SEED);

		// transform data to normal distribution: -------------------------
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(allData);
		normalizer.transform(allData);

		// split data to test and train sets: -----------------------------
		//SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(PERCENT_TRAIN);
		//DataSet trainingData = testAndTrain.getTrain();
		//DataSet testData = testAndTrain.getTest();
		return allData;
	}

	protected static void plotDataWithJzy3d(MultiLayerNetwork model, DataSet dataset, float pointWidth) {

		// surface: ---------------------------------------
		
		// Take the classifier output as the function to plot:
		Mapper mapper = new Mapper() {
			@Override 
			public double f(double x, double y) {
				double [][]  data  =  {  { x , y}, { }  };                
				INDArray output = model.output(new NDArray(data));
				return output.getDouble(0);
			} 
		}; 

		// Define range and precision for the function to plot 
		Range range = new Range(-3, 3);
		int steps = 120;

		// scatter: ---------------------------------------
		
		// draw the input and the labels with color coding:
		INDArray input = dataset.getFeatures();
		INDArray labels = dataset.getLabels();

		int size = input.size(0);

		float x, y, z;

		Coord3d[] points = new Coord3d[size];
		Color[]   colors = new Color[size];

		for(int i = 0; i < size; i++){
			
			x = input.getFloat(i, 0);
			y = input.getFloat(i, 1);
			z = labels.getFloat(i, 0);

			points[i] = new Coord3d(x, y, z);
			
			// set the color by comparing the network output to the matching label:
			// RED:    1 classified as 1
			// YELLOW: 1 classified as 0
			// BLUE:   0 classified as 0
			// GREEN:  0 classified as 1
			colors[i] = mapper.f(x, y) > 0.5 ? (z > 0.5 ? Color.RED : Color.GREEN) : (z > 0.5 ? Color.YELLOW : Color.BLUE);
		} 


		// combine surface and scatter into one plotter:
		CombinedPlotter plotter = new CombinedPlotter(mapper, range, steps, points, colors, pointWidth);

		try {
			plotter.plot();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
