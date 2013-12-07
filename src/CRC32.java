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
            System.out.print(fileData);
            System.out.print("\n");
            byte[] bytes = fileData.getBytes();
            java.util.zip.CRC32 x = new java.util.zip.CRC32();
            x.update(bytes);
            System.out.println("CRC32 (via Java's library)     = " + Long.toHexString(x.getValue()));
        }catch(Exception e){
            System.out.println(e.getMessage());
        }
    }

}


