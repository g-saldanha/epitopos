
OUTPUT_Epitopos="log_epitopos.txt"

BASIC_PATH="./"

################################################### DESC PATHS ##############################################################

DESC_EPITOPOS=$BASIC_PATH"./epitopos.json"

##################################################################################################################################
#################################################### RUN EPITOPOS ################################################################
##################################################################################################################################

java -Xmx25000M -jar "./Json_MASTERMovelets.jar" -curpath "$BASIC_PATH" -respath "$BASIC_PATH" -descfile "$DESC_EPITOPOS"  -nt 6 -ed true -ms 1 -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" -pvt true -pp 10 -lp false -op false | tee -a "$OUTPUT_Epitopos"   
wait

# -nt Number of Threads (-nt 6)
# -ed Explore Dimensions (-ed true)
# -ms Minimun Size of movelet (-ms 1)
# -cache Can it use your cache? (-cache true)
# -output Type of output table (-output discrete)
# -samples 1 -sampleSize 0.5 -medium "none"  -lowm "false" (Não mexer)
# -pvt If you will use the pivots or not (-pvt true) 
# -pp Porcentage of points that will become pivots (-pp 10)
# -lp Last Prunning (-lp false)
# -op If only the pivots will become movelets (-op false) 
# -nt 6 -ed true -ms 1 -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" -pvt true -pp 10 -lp false -op false | tee -a "$OUTPUT_MASTERMovelets_log"   
