package testjfrog;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;

/**
 * Test loading a Tensorflow classifier using the jdl4pam library. 
 * @author Jamie Macaulay
 *
 */
public class TestJFrog {
	
	public static void main( String[] args ) {
		args = new String[1]; 
		args[0] = "C:\\Users\\Jamie\\Desktop\\Right_whales_DG\\model_lenet_dropout_input_conv_all\\saved_model.pb"; 
		String modelPath = args[0]; 
		File file = new File(modelPath);
		Path modelDir = Paths.get(file.getAbsoluteFile().getParent()); 
		String modelName = file.getName(); 
	
		System.out.println("Loading model: " + modelName);
		
		System.out.println(Engine.getAllEngines()); 

		Model model = Model.newInstance(modelPath, "TensorFlow");
		try {
			model.load(modelDir, "saved_model.pb");
			System.out.println("Model loaded successfully: " + modelName);
		} catch (MalformedModelException | IOException e) {
			e.printStackTrace();
		}
	}
}
