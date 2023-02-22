package nyu.edu.bigdata;


import java.io.IOException;
import java.util.*;

// hadoop runtime dependencies
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
public class nGramProb {

    // this is the driver that runs all the jobs
    public static void main(String[] args) throws Exception {

        // get a reference to a job runtime configuration for this program
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        //job 1 as th word count
        Job count_word = Job.getInstance(conf, "word count");
        count_word.setJarByClass(nGramProb.class);
        count_word.setMapperClass(MyMapper1.class);
        count_word.setReducerClass(MyReducer1.class);
        count_word.setOutputKeyClass(Text.class);
        count_word.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(count_word, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(count_word, new Path(otherArgs[1] +"/counts"));
        count_word.waitForCompletion(true);


        //wait for job1 to complete and get the ngram counters
        Counter uniCount = count_word.getCounters().findCounter(CountersClass.UNIGRAM.COUNT_UNIGRAM);
        Counter biCount = count_word.getCounters().findCounter(CountersClass.BIGRAM.COUNT_BIGRAM);
        Counter triCount = count_word.getCounters().findCounter(CountersClass.TRIGRAM.COUNT_TRIGRAM);


        //job2 to count ngram word probabilities
        Configuration conf2 = new Configuration();
        Job ngram_prob = Job.getInstance(conf2, "ngram probability count");

        //set the hadoop counters received in the second job
        ngram_prob.getConfiguration().setLong(CountersClass.UNIGRAM.COUNT_UNIGRAM.name(), uniCount.getValue());
        ngram_prob.getConfiguration().setLong(CountersClass.BIGRAM.COUNT_BIGRAM.name(), biCount.getValue());
        ngram_prob.setJarByClass(nGramProb.class);
        ngram_prob.setMapperClass(MyMapper2.class);
        ngram_prob.setReducerClass(MyReducer2.class);
        ngram_prob.setOutputKeyClass(Text.class);
        ngram_prob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(ngram_prob, new Path(otherArgs[1]+"/counts"));
        FileOutputFormat.setOutputPath(ngram_prob, new Path(otherArgs[1]+"/probability"));
        ngram_prob.waitForCompletion(true);


        //job3 to count united_states probabilities
        Configuration conf3 = new Configuration();
        Job prob_US = Job.getInstance(conf3, "united states probability count");
        prob_US.getConfiguration().setLong(CountersClass.TRIGRAM.COUNT_TRIGRAM.name(), triCount.getValue());
        prob_US.setJarByClass(nGramProb.class);
        prob_US.setMapperClass(MyMapper3.class);
        prob_US.setReducerClass(MyReducer3.class);
        prob_US.setOutputKeyClass(Text.class);
        prob_US.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(prob_US, new Path(otherArgs[1]+"/counts"));
        FileOutputFormat.setOutputPath(prob_US, new Path(otherArgs[1]+"/unitedstates"));
        System.exit(prob_US.waitForCompletion(true) ? 0 : 1);

    }

    //used the concept of hadoop counters to track unique ngrams
    public static class CountersClass {
        public static enum UNIGRAM {
            COUNT_UNIGRAM
        }
        public static enum BIGRAM {
            COUNT_BIGRAM
        }
        public static enum TRIGRAM {
            COUNT_TRIGRAM
        }
    }

    // this class implements the mapper for word count
    public static class MyMapper1 extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text uni_gram = new Text();
        private Text bi_gram = new Text();
        private Text tri_gram = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            String each_lines = value.toString();
            each_lines = each_lines.toLowerCase();
            each_lines = each_lines.replaceAll("[^A-Za-z0-9]", " ");
            String[] sentences = each_lines.split("\n");

            for (int i=0; i<sentences.length; ++i){
                String[] word_tokens = sentences[i].split("\\s+");
                if (word_tokens.length < 3) {
                    continue;
                }

                //running uni_grams
                for (int j=0; j<word_tokens.length; ++j) {
                    uni_gram.set(word_tokens[j]);
                    context.write(uni_gram, one);
                }

                //running bi_grams
                for (int j=0; j<word_tokens.length-1; ++j) {
                    bi_gram.set(word_tokens[j] + " " + word_tokens[j + 1]);
                    context.write(bi_gram, one);
                }

                //running tri_grams
                for (int j=0; j<word_tokens.length-2; ++j){
                    tri_gram.set(word_tokens[j] + " " + word_tokens[j+1] + " " + word_tokens[j+2]);
                    context.write(tri_gram, one);
                }


            }
        }
    }

    // this class implements the reducer for the word count
    public static class MyReducer1 extends Reducer<Text,IntWritable,Text,IntWritable> {

        private IntWritable final_result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int total = 0;
            int key_len = key.toString().split("\\s+").length;


            for (IntWritable val : values) {
                total += val.get();
            }
            //hadoop uni_gram counter
            if(key_len == 1) {
                context.getCounter(CountersClass.UNIGRAM.COUNT_UNIGRAM).increment(total);
            }
            //hadoop bi_gram counter
            else if (key_len ==2) {
                context.getCounter(CountersClass.BIGRAM.COUNT_BIGRAM).increment(total);
            }
            else {
                context.getCounter(CountersClass.TRIGRAM.COUNT_TRIGRAM).increment(total);
            }
            final_result.set(total);
            context.write(key, final_result);
        }
    }

    public static class MyMapper2 extends Mapper<Object, Text, Text, Text> {

        //private final static IntWritable one = new IntWritable(1);

        //private Text word = new Text(
        private Text uni_gram = new Text();
        private Text bi_gram = new Text();
        private Text output_UNIGRAM = new Text();
        private Text output_BIGRAM  = new Text();


        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            String lines = value.toString();

            String[] line = lines.split("\n");

            for (int i=0; i<line.length; ++i){
                String[] words = line[i].split("\\s+");
                int len = words.length;

                if (len == 2) {
                    uni_gram.set("unigram");
                    output_UNIGRAM.set(words[0] + "\t" + words[1]);
                    context.write(uni_gram, output_UNIGRAM);
                }
                else if (len == 3) {
                    bi_gram.set("bigram");
                    output_BIGRAM.set(words[0] + " " + words[1] + "\t" + words[2]);
                    context.write(bi_gram, output_BIGRAM);
                }

            }

        }
    }

    public static class MyReducer2 extends Reducer<Text,Text,Text, DoubleWritable> {

        private DoubleWritable final_result = new DoubleWritable();
        private Text output_Gram = new Text();
        private Map<String, Integer> bigramCounts;
        private Map<String, Integer> unigramCounts;
        private double uniGram_total;
        private double biGram_total;



        //getting the global hadoop counters in the reducer after job1 is finsihed
        @Override
        public void setup(Context context) throws IOException, InterruptedException{
            super.setup(context);
            this.uniGram_total  = context.getConfiguration().getLong(CountersClass.UNIGRAM.COUNT_UNIGRAM.name(), 0);
            this.biGram_total  = context.getConfiguration().getLong(CountersClass.BIGRAM.COUNT_BIGRAM.name(), 0);
            unigramCounts = new HashMap<>();
            bigramCounts = new HashMap<>();
            System.out.println(uniGram_total);
            System.out.println(biGram_total);

        }
        public void reduce(Text key, Iterable<Text> values,
                           Context context) throws IOException, InterruptedException {

            for (Text var : values) {
                String[] tempArray = var.toString().split("\t");
                if(key.toString().equals("unigram")) {

                    unigramCounts.put(tempArray[0], Integer.valueOf(tempArray[1]));

                }
                else if(key.toString().equals("bigram")) {

                    bigramCounts.put(tempArray[0], Integer.valueOf(tempArray[1]));

                }

            }

        }
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (Map.Entry<String, Integer> entry : bigramCounts.entrySet()) {
                String[] biword = entry.getKey().toString().split("\\s+");
                int uniw1 = unigramCounts.get(biword[0]);
                double bigramCount = entry.getValue();
                double prob = (double) (bigramCount / biGram_total )/ (uniw1 / uniGram_total);
                if (prob > 1){
                    prob =1;
                }

                final_result.set(prob);
                output_Gram.set(biword[0] + " " + biword[1]);
                context.write(output_Gram,final_result);
            }
        }
    }
    public static class MyMapper3 extends Mapper<Object, Text, Text, Text> {


        private Text trigram3 = new Text();
        private Text output_trigram3 = new Text();


        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {


            //this mapper outputs each word containing united_states for the reducer
            String each_line = value.toString();
            String[] word_split = each_line.split("\n");

            for (int i=0; i<word_split.length; ++i){
                String[] words = word_split[i].split("\\s+");
                int len = words.length;

                if (len == 4){
                    //checking first two word as united_states

                    if ("united".equals(words[0]) && "states".equals(words[1]))
                    {
                        trigram3.set("united states");
                        output_trigram3.set(words[2] + "\t" + words[3]);
                        context.write(trigram3, output_trigram3);
                    }
                }

            }

        }
    }

    public static class MyReducer3 extends Reducer<Text,Text,Text, DoubleWritable> {

        private DoubleWritable result = new DoubleWritable();
        private Text keyGram = new Text();
        private double triGram_total;
        @Override
        public void setup(Context context) throws IOException, InterruptedException{
            super.setup(context);
            this.triGram_total  = context.getConfiguration().getLong(CountersClass.TRIGRAM.COUNT_TRIGRAM.name(), 0);
            System.out.println("total trigrams = "+triGram_total);
        }
        public void reduce(Text key, Iterable<Text> values,
                           Context context) throws IOException, InterruptedException {

            //maximum probability variable
            double max_prob =0;

            //maximum probability word variable
            String final_word = null;
            //traversing each tri grams individually

            for (Text var : values) {

                //converting each trigram to string
                String each_Word = var.toString();

                //storing it in a temporary string array
                String[] inputArray = each_Word.split("\t");

                for(int i =0; i< inputArray.length; i+=2)
                {
                    if(Double.parseDouble(inputArray[i+1]) > max_prob ) {
                        max_prob = Double.parseDouble(inputArray[i+1]);
                        final_word = inputArray[i];
                    }
                }
            }
            keyGram.set(final_word);
            result.set(max_prob/triGram_total);
            context.write(keyGram, result);
        }
    }


}