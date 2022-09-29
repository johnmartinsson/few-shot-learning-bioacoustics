for i in Evaluation_Set_22050Hz/*/*/*.wav; 
do
        sox "$i" -r 22050 tmp.wav;
        mv tmp.wav "$i"
done

