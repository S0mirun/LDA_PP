module lat_lon
    implicit none
    double precision, parameter :: pi  = 4.0d0 * datan(1.0d0)
    double precision, parameter :: RD   = 180.0d0 / pi
    double precision, parameter :: DR   = pi / 180.0d0
contains
    subroutine CalDistCo(ALong,ALat,Dist,Co)
        !****************************************************************************    
        !  compute disntance traveled and course of ship by of 2 latitude and longitude of points using 漸長緯度航法
        !  input are DICIMAL DEGREE (ddd.dddd)
        !
        !   Input variables
        !   ALong: Longitude vector of two points [start end](deg.)
        !   ALat:  Latitude vector of two point [start end](deg.)
        !
        !   Output variables
        !   dist: distance traveled（min, to convert to meter, *1852）
        !   co: course (degree, north = 0)
        !
        !****************************************************************************
            implicit none
            double precision,INTENT(IN) :: ALong(2),ALat(2)
            DOUBLE PRECISION,INTENT(OUT) :: Dist, CO
            double precision :: AM1,AM2,AMD,ALongD,ALatD,ALatM,ADep
            !
            ! Calculating the Earth as a sphere
            !  compute 
            AM1=7915.7045D0*log10(tan((45.0D0+ALat(1)/2.0D0)*DR))
            AM2=7915.7045D0*log10(tan((45.0D0+ALat(2)/2.0D0)*DR))
            !  compute meri-dional parts (unit: min)
            AMD=AM2-AM1
            !  compute latidude diff (unit: degree)
            ALatD=ALat(2)-ALat(1)
            !  compute longitude diff (unit:min)
            ALongD=(ALong(2)-ALong(1))*60.0D0
            ! compute middle latitude
            ALatM=(ALat(2)+ALat(1))/2.0D0
            ! compute touzaikyo(departure)
            ADep=ALongD*cos(ALatM*DR)
            ! compute course by mean middle latitude sailing
            if (abs(ADep) <= epsilon(1.0d0) .and. abs(ALatD) <= epsilon(1.0d0))then
                co = 0.0d0
            else
                co = atan2(ADep/60.0D0, ALatD)
            endif
            if (co > 2.0d0 * pi)then
                co = co-2.0d0*pi
            elseif (co < 0.0d0)then
                co = co + 2.0d0*pi
            endif
            
            Co=Co*RD
            
            !!! COMPUTE traveled distance
            ! around 90 deg and 270 deg, use middle latitude sailing    
            if ( co >= 89.0D0 .and. co <= 91.0D0)then
                dist = sqrt( ( ALatD * 60.0D0 ) * ( ALatD * 60.0D0 ) + ( ADep ) * ( ADep ) )
            elseif (co >= 269.0D0 .and. co <= 271.0D0)then
                dist = sqrt( ( ALatD * 60.0D0 ) * ( ALatD * 60.0D0 ) + ( ADep ) * ( ADep ) )
            else
                !  use mercator's method to compute course
                if (abs(ALongD) <= epsilon(1.0d0) .and. abs(AMD) <= epsilon(1.0d0))then
                    co =0.0d0
                else
                    co =  atan2( ALongD , AMD  )
                endif
                
                if (co > 2.0d0*pi)then
                    co = co-2.0d0*pi
                elseif (co < 0.0d0)then
                    co = co + 2.0d0*pi
                endif
                co = co * RD
                ! compute traveled distance
                dist = abs( ( ALatD ) * 60.0D0 / cos( co * DR ) )
            endif 
        end subroutine CalDistCo
        
        ! %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    
        ! %  Calculate Lat and Lon for given Dist and co
        ! %  input are DICIMAL DEGREE (ddd.dddd) and nautical mile for Dist
        ! %
        ! %   Input variables
        ! %   lat_origin: Longitude of origin (deg.)
        ! %   lon_origin: latitude of origin (deg.)
        ! %   Dist      : traveled distance of given point from origin (nm)
        ! %   co        : cource of given point, north=0, radian
        ! %
        ! %   Output variables
        ! %   lat_output: Longitude of given point (deg.)
        ! %   lon_output: latitude of given point (deg.)
        ! %
        ! %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    subroutine SailingCal(lat_origin,lon_origin,lat_output,lon_output,Dist,co)
        implicit none
        integer :: domain
        DOUBLE PRECISION, INTENT(IN):: lat_origin,lon_origin,Dist,co
        DOUBLE PRECISION, INTENT(OUT):: lat_output, lon_output
        DOUBLE PRECISION:: AMA0,AMA1,MDLat,Dep,MLat,DLat,DLon, co_mod
        !
        if (co >= 2.0d0 * pi) then
            co_mod = co-2 * pi
        elseif (co < 0) then
            co_mod = co + 2.0d0 * pi
        else 
            co_mod = co
        endif
        !   Determine the domain in whcih co is existing
        if(mod(co_mod,2.0d0 * pi)>=0 .and. mod(co_mod,2.0d0 * pi)<pi/2)then
            domain =1 
        elseif(mod(co_mod,2.0d0 * pi)>=pi/2.0d0 .and. mod(co_mod,2.0d0 * pi)<pi)then
            domain =2 
        elseif(mod(co_mod,2.0d0 * pi)>=pi .and. mod(co_mod,2.0d0 * pi)<pi*3.0d0/2.0d0)then
            domain =3 
        elseif(mod(co_mod,2.0d0 * pi)>=pi*3.0d0/2.0d0 .and. mod(co_mod,2.0d0 * pi)<2.0d0 * pi)then
            domain =4 
        endif
        if(abs(co_mod-pi/2.0d0)>=1.0d0 .and. (abs(co_mod-3.0d0/2.0d0 * pi)>=1.0d0))then
            ! Meridional part at lat_origin
            AMA0 =7915.7045D0*log10(tan(pi/180.0d0*(45.0d0+lat_origin/2.0D0)))
            DLat =abs(Dist*cos(co_mod)/60.0d0)  ! in degree
            
            if(domain==1 .or. domain==4)then
                DLat= DLat
            elseif(domain==2 .or. domain==3)then
                DLat=-DLat
            endif
            lat_output =lat_origin+DLat ! in degree
            
            AMA1 =7915.7045D0*log10(tan(pi/180.0d0*(45.0D0+lat_output/2.0D0)))
            MDLat=AMA1-AMA0
            DLon =abs(MDLat*tan(co_mod)/60.0d0) ! in degree
            
            if(domain==1 .or. domain==2)then
                DLon= DLon
            elseif(domain==3 .or. domain==4)then
                DLon=-DLon
            endif
            lon_output =lon_origin+DLon
        else
            !  Mean Middle Latitude Sailing
            DLat=abs(Dist*cos(co_mod)/60.0d0)  ! in degree
            Dep =abs(Dist*sin(co_mod)/60.0d0)  ! in degree
            if(domain==1 .or. domain==4)then
                DLat= DLat
            elseif(domain==2 .or. domain==3)then
                DLat=-DLat
            endif
            lat_output=lat_origin+DLat
            MLat=(lat_origin+lat_output)*0.5d0
            DLon=abs(Dep/cos(pi/180.0d0*(MLat)))        !in degree
            if(domain==1 .or. domain==2)then
                DLon= DLon
            elseif(domain==3 .or. domain==4)then
                DLon=-DLon
            endif
            lon_output =lon_origin+DLon
        endif
        end subroutine SailingCal
    end module lat_lon