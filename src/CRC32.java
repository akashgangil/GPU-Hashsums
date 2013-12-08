/*JAVA CRC32 implementation using MPEG Polynomial*/

import java.io.*;
import java.util.Scanner;

public class CRC32 { 


    private static String readFile(String pathname) throws IOException {

        InputStream fstream = new FileInputStream("input.in");
        DataInputStream in = new DataInputStream(fstream);
        BufferedReader br = new BufferedReader(new InputStreamReader(in));
        String strLine = br.readLine();
        return strLine.substring(0, strLine.length()-1);
    }


    public static void main(String[] args) { 
        try{
            String fileData = readFile("input.in");
            byte[] bytes = fileData.getBytes();
            java.util.zip.CRC32 x = new java.util.zip.CRC32();
            long start_timer = System.currentTimeMillis();
            x.update(bytes);
            long end_timer = System.currentTimeMillis();           
            System.out.println("CRC32 (via Java's library)     = " + Long.toHexString(x.getValue()));
            System.out.println(end_timer - start_timer + "ms");
        }catch(Exception e){
            System.out.println(e.getMessage());
        }
    }

}


