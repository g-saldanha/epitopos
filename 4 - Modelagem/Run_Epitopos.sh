
OUTPUT_Epitopos_Full="log_epitopos_full.txt"
OUTPUT_Epitopos_Without_Bepipred="log_epitopos_without_bepipred.txt"
OUTPUT_Epitopos_Without_Bepipred_and_Aminoacid="log_epitopos_without_bepipred_and_aminoacid.txt"
OUTPUT_Epitopos_Full_With_LP="log_epitopos_full_with_last_pruning.txt"

##################################################################################################################################
#################################################### RUN EPITOPOS ################################################################
##################################################################################################################################

#java  -Xmx50000M -jar "./JSON_MASTER_Pivots.jar" -curpath "./" -respath "./" -descfile "./full_epitopos.json"  -nt 10 -ed true -ms 1 -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" -pvt true -pp 10 -lp false -op false | tee -a "$OUTPUT_Epitopos_Full"
#wait

#java  -Xmx50000M -jar "./JSON_MASTER_Pivots.jar" -curpath "./" -respath "./" -descfile "./epitopos_without_bepipred.json"  -nt 10 -ed true -ms 1 -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" -pvt true -pp 10 -lp false -op false | tee -a "$OUTPUT_Epitopos_Without_Bepipred"
#wait

java  -Xmx50000M -jar "./JSON_MASTER_Pivots.jar" -curpath "./" -respath "./" -descfile "./full_epitopos.json"  -nt 10 -ed true -ms 1 -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" -pvt true -pp 10 -lp true -op false | tee -a "log_epitopos_full_with_last_pruning.txt"
#wait


#java  -Xmx50000M -jar "./JSON_MASTER_Pivots.jar" -curpath "./" -respath "./" -descfile "./epitopos_without_bepipred_and_aminoacid.json"  -nt 10 -ed true -ms 1 -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" -pvt true -pp 10 -lp false -op false | tee -a "$OUTPUT_Epitopos_Without_Bepipred_and_Aminoacid"
#wait

# -nt Number of Threads (-nt 6)
# -ed Explore Dimensions (-ed true)
# -ms Minimun Size of movelet (-ms 1)
# -cache Can it use your cache? (-cache true)
# -output Type of output table (-output discrete)
# -samples 1 -sampleSize 0.5 -medium "none"  -lowm "false" (N�o mexer)
# -pvt If you will use the pivots or not (-pvt true) 
# -pp Porcentage of points that will become pivots (-pp 10)
# -lp Last Prunning (-lp false) [MUDE PARA true PARA FAZER O TESTE E VER NO QUE DA]
# -op If only the pivots will become movelets (-op false) 
# -nt 6 -ed true -ms 1 -cache true -output discrete -samples 1 -sampleSize 0.5 -medium "none" -output "discrete" -lowm "false" -pvt true -pp 10 -lp false -op false | tee -a "$OUTPUT_MASTERMovelets_log"   
