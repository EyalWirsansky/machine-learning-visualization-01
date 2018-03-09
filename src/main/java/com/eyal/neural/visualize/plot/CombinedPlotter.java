package com.eyal.neural.visualize.plot;

import java.util.Random;

import org.jzy3d.analysis.AbstractAnalysis;
import org.jzy3d.analysis.AnalysisLauncher;
import org.jzy3d.chart.factories.AWTChartComponentFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.maths.Coord3d;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Builder;
import org.jzy3d.plot3d.builder.Mapper;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Scatter;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.rendering.canvas.Quality;
 
// a combination of surface and scatter plots on the same axis system
public class CombinedPlotter extends AbstractAnalysis { 
 
	// surface:
	private final Mapper mapper;
	private final Range range;
	private final int steps;
	
	// scatter:
	private final Coord3d[] points;
	private final Color[]   colors;
	private final float pointWidth;
	
    public CombinedPlotter(Mapper mapper, Range range, int steps, Coord3d[] points, Color[] colors, float pointWidth) {
		super();
		this.mapper = mapper;
		this.range = range;
		this.steps = steps;
		
		this.points = points;
		this.colors = colors;
		this.pointWidth = pointWidth;
	}

	@Override 
    public void init() { 
        // surface:
        final Shape surface = Builder.buildOrthonormal(new OrthonormalGrid(range, steps, range, steps), mapper);
        surface.setColorMapper(new ColorMapper(new ColorMapRainbow(), surface.getBounds().getZmin(), surface.getBounds().getZmax(), new Color(1, 1, 1, .5f)));
        surface.setFaceDisplayed(true);
        surface.setWireframeDisplayed(false);
        surface.setWireframeColor(Color.BLACK);
        chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType()); 
        chart.getScene().getGraph().add(surface);
 
        // scatter:
        Scatter scatter = new Scatter(points, colors, pointWidth);        
        chart.getScene().add(scatter);
    } 
    
    public void plot() throws Exception { 
    	AnalysisLauncher.open(this); 
    }
    
    public static void main(String[] args) throws Exception {
        // surface: -------------------
        Mapper mapper = new Mapper() {
            @Override 
            public double f(double x, double y) {
                return x * Math.sin(x * y);
            } 
        }; 
 
        // Define range and precision for the function to plot 
        Range range = new Range(-3, 3);
        int steps = 80;

        // scatter: -------------------
        int size = 50000;
        float x;
        float y;
        float z;
        float a;
         
        Coord3d[] points = new Coord3d[size];
        Color[]   colors = new Color[size];
         
        Random r = new Random();
        r.setSeed(0);
         
        for(int i=0; i<size; i++){
        	x = r.nextFloat() * (range.getMax() - range.getMin()) - range.getMax() ;
        	y = r.nextFloat() * (range.getMax() - range.getMin()) - range.getMax() ;
        	z = r.nextFloat() * (range.getMax() - range.getMin()) - range.getMax() ;

            points[i] = new Coord3d(x, y, z);
            a = 0.25f;
            colors[i] = new Color(x, y, z, a);
        } 
        
        // draw:
        CombinedPlotter plotter = new CombinedPlotter(mapper, range, steps, points, colors, 2.0f);
        plotter.plot();
    } 
}  