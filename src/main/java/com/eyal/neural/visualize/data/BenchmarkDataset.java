package com.eyal.neural.visualize.data;

public enum BenchmarkDataset {
	IRIS_DATASET("iris.txt", 4, 3, 150),
	IRIS_2D_DATASET("iris_2D.txt", 2, 2, 100),
	CLOUDS_DATASET("clouds.txt", 2, 2, 5000),
	CONCENTRIC_DATASET("concentric.txt", 2, 2, 2500),
	GAUSS_2D_DATASET("gauss_2D.txt", 2, 2, 5000);
	
	final public String fileName;
	final public int featuresCount;
	final public int classesCount;
	final public int fileLines;

	private BenchmarkDataset(String fileName, int featuresCount, int classesCount, int fileLines) {
		this.fileName = fileName;
		this.featuresCount = featuresCount;
		this.classesCount = classesCount;
		this.fileLines = fileLines;
	}
}