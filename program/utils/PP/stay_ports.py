"""
着桟する港のlat,lonを保存するファイル
新しいclassを追加したら、ALL = [] にも追加する
"""

class Sea:
    class OSAKA:
        minY_lat = 34.292719
        maxY_lat = 34.753652
        minY_long = 134.903880
        maxY_long = 135.474781
        name = '__Osaka_Bay'

    class IMABARI:
        minY_lat = 33.436031
        maxY_lat = 34.903841
        minY_long = 132.292302
        maxY_long = 134.098745
        name = '__Setouchi'

    class SHIMONOSEKI:
        minY_lat = 33.762043
        maxY_lat = 34.240051
        minY_long = 130.724063
        maxY_long = 131.313142
        name = '__Shimonoseki'
    
    ALL = [OSAKA, IMABARI, SHIMONOSEKI]


class Hokkaido:
    class tomakomai:
        minY_lat = 42.579252
        maxY_lat = 42.678146
        minY_long = 141.570496
        maxY_long = 141.707653
        name = 'Tomakomai'
        num = "01"

    class kushiro:
        minY_lat = 42.934237
        maxY_lat = 43.054574
        minY_long = 144.268750
        maxY_long = 144.436638
        name = 'Kushiro'
        num = "01"

    class ishikari_bay:
        minY_lat = 43.159633
        maxY_lat = 43.262626
        minY_long = 141.219361
        maxY_long = 141.361946
        name = 'Otaru'
        num = "01"

    class hakodate_bay:
        minY_lat = 41.775662
        maxY_lat = 41.817065
        minY_long = 140.672654
        maxY_long = 140.731750
        name = 'Hakodate'
        num = "01"
    
    ALL = [tomakomai, kushiro, ishikari_bay, hakodate_bay]

class Honsyu:
    class Tohoku:
        class hachinohe:
            minY_lat = 40.515826
            maxY_lat = 40.560259
            minY_long = 141.512348
            maxY_long = 141.571779
            name = 'Hachinohe'
            num = "02"

        class aomori_bay:
            minY_lat = 40.831995
            maxY_lat = 40.895291
            minY_long = 140.713490
            maxY_long = 140.797928
            name = 'Aomori'
            num = "02"

        class akita:
            minY_lat = 39.720417
            maxY_lat = 39.814846
            minY_long = 139.952722
            maxY_long = 140.077115
            name = 'Akita'
            num = "05"

        class isinomaki_bay:
            minY_lat = 38.274610
            maxY_lat = 38.476989
            minY_long = 141.179720
            maxY_long = 141.441471
            name = 'Ishinomaki'
            num = "04"

        class onahama:
            minY_lat = 36.909405
            maxY_lat = 36.947614
            minY_long = 140.868994
            maxY_long = 140.920306
            name = 'Onahama'
            num = "07"

        class iwate:
            minY_lat = 39.235741
            maxY_lat = 39.276861
            minY_long = 141.880832
            maxY_long = 141.934787
            name = 'Kamaishi'
            num = "03"
        
        ALL = [hachinohe, aomori_bay, akita, isinomaki_bay, onahama, iwate]
    
    class nihonkai:
        class nigata:
            minY_lat = 37.178366
            maxY_lat = 37.202082
            minY_long = 138.240258
            maxY_long = 138.270778
            name = 'Nigata'
            num = "15"

        class kanazawa:
            minY_lat = 36.606281
            maxY_lat = 36.633193
            minY_long = 136.591621
            maxY_long = 136.627522
            name = 'Kanazawa'
            num = "17"

        class tottori:
            minY_lat = 35.508565
            maxY_lat = 35.566512
            minY_long = 133.229988
            maxY_long = 133.306708
            name = 'Sakaiminato'        
        
        ALL = [nigata, kanazawa, tottori]

    class Pasific:
        class ibaragi:
            minY_lat = 35.910456
            maxY_lat = 35.958665
            minY_long = 140.667261
            maxY_long = 140.727823
            name = 'Kashima'
            num = "08"

        class tokyo_bay:
            minY_lat = 35.200483
            maxY_lat = 35.689946
            minY_long = 139.595564
            maxY_long = 140.235517
            name = 'Tokyo'
            num = "12"

        class suruga_bay:
            minY_lat = 34.981367
            maxY_lat = 35.083268
            minY_long = 138.487049
            maxY_long = 138.613881
            name = 'Shimizu'
            num = "22"

        class ise_bay:
            minY_lat = 34.941052
            maxY_lat = 34.972875
            minY_long = 136.635710
            maxY_long = 136.671490
            name = 'Yokkaichi' 
            num = "24"   

        class osaka_bay:
            minY_lat = 34.571811
            maxY_lat = 34.607731
            minY_long = 135.401341
            maxY_long = 135.445586
            name = 'Osaka_port1A'
            num = "27"

        class osaka_bay2:
            minY_lat = 34.653702
            maxY_lat = 34.685382
            minY_long = 135.176362
            maxY_long = 135.217267
            name = 'Kobe'
            num = "28"

        class nanko:
            minY_lat = 34.384388
            maxY_lat = 34.423418
            minY_long = 135.195461
            maxY_long = 135.243042
            name = 'KIX'
            num = "27"

        class setouchi:
            minY_lat = 34.342116
            maxY_lat = 34.363660
            minY_long = 133.828336
            maxY_long = 133.854798
            name = 'Sakaide'

        class setouchi2:
            minY_lat = 34.216552
            maxY_lat = 34.275350
            minY_long = 132.487686
            maxY_long = 132.559464
            name = 'Kure'        
            
        class kagoshima_bay:
            minY_lat = 31.495868
            maxY_lat = 31.515861
            minY_long = 130.517674
            maxY_long = 130.541737
            name = 'kagoshima'
        
        ALL = [ibaragi, tokyo_bay, 
               suruga_bay, ise_bay, 
               osaka_bay, osaka_bay2, nanko, 
               setouchi, setouchi2, 
               kagoshima_bay
               ]

    ALL = Tohoku.ALL + nihonkai.ALL + Pasific.ALL