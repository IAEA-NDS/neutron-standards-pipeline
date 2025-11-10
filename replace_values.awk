#!/usr/bin/gawk -f
#
# GAWK script to extract estimated cross sections
# from a GMAP result file (typically called dat.res)
# and write them into a DAT.INP file (used by DATP
# for data reduction).
#
# Call:
#   gawk -f replace_values.awk gma.res DAT.INP > DAT_NEW.INP


# Helper function to trim whitespaces from
# beginning and end of a string.
function trim(s) {
    gsub(/^[ \t\r]+|[ \t\r]+$/, "", s)
    return s
}

# We capture each reaction the result 
# recorded in the gma.res file to be
# able to write them into the DAT.INP file
# afterwards.
ARGIND == 1 && /^1   RESULT/ {
    reac = trim(substr($0, 11))
    getline
    getline
    getline
    getline
    
    idx = 0
    while (1) {
        getline
        if (length($0) <= 66) {
            break
        }
        idx = idx + 1
        reac_arr[reac][idx]["en"] = $1
        reac_arr[reac][idx]["xs"] = $2
    }
}

ARGIND == 2 {
    line = trim($0)
    if (line in reac_arr) {
        print " " line
        # for the first energy in DAT.INP, we use the
        # cross section of the first energy extracted
        # from gma.res.
        getline
        cur_en = substr($0, 0, 10)
        cur_xs = reac_arr[line][1]["xs"]
        printf "%s%10.4E\n", cur_en, cur_xs
        # Afterwards we go through the list of 
        # (energy, cross section) pairs extracted from
        # gma.res, check whether the energies match with
        # those in DAT.INP and if so, replace the cross sections
        # in DAT.INP by those from gma.res
        num_els = length(reac_arr[line])
        for (i = 1; i <= num_els; i++) {
            getline
            cur_en = substr($0, 0, 10)
            cur_xs = substr($0, 10, 10)
            ref_en = reac_arr[line][i]["en"]
            ref_xs = reac_arr[line][i]["xs"]
            # +0 to be sure that numbers rather than strings are compared
            if (cur_en+0 != ref_en+0) {
                print("ERROR: ENERGY MISMATCH") > "/dev/stderr"
                exit 1
            }
            printf "%s%10.4E\n", cur_en, ref_xs
        }
        # We expect one more line in DAT.INP for which we use
        # the last read cross section from gma.res
        getline
        cur_en = substr($0, 0, 10)
        cur_xs = reac_arr[line][num_els]["xs"]
        printf "%s%10.4E\n", cur_en, cur_xs
    } else {
        print $0
    }
}
