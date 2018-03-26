# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:40:28 2017
Routine to particle track through ROMS's climatological fields

@author: Xiajian
Modified by JiM in Feb 2018 w/more documentation
Simplified and modified by huimin in Mar 2018 for Cape Cod Bay
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime, timedelta


def nearlonlat(lon,lat,lonp,latp):
    """
    i=nearlonlat(lon,lat,lonp,latp)
    find the closest node in the array (lon,lat) to a point (lonp,latp)
    input:
    lon,lat - np.arrays of the grid nodes, spherical coordinates, degrees
    lonp,latp - point on a sphere
    output:
    i - index of the closest node
    min_dist - the distance to the closest node, degrees
    For coordinates on a plane use function nearxy
    
    Vitalii Sheremet, FATE Project
    """
    cp=np.cos(latp*np.pi/180.)
    # approximation for small distance
    dx=(lon-lonp)*cp
    dy=lat-latp
    dist2=dx*dx+dy*dy
    # dist1=np.abs(dx)+np.abs(dy)
    
    i=np.argmin(dist2)
        
    #    min_dist=np.sqrt(dist2[i])
    return i 
def rot2d(x, y, ang):
    '''rotate vectors by geometric angle'''
    xr = x*np.cos(ang) - y*np.sin(ang)
    yr = x*np.sin(ang) + y*np.cos(ang)
    return xr, yr
def find_nearest_coast(lon,lat,CL):
    '''get index of nearest point on coastline'''
    dist=[]
    for i in range(len(CL)):
        #dis=[]
        d=(lon-CL[i][0])**2+(lat-CL[i][1])**2
        #print 'dis_nearest:',d
        #dis.append(i)
        #dis.append(d)
        dist.append(d)
    index_nearest=np.argmin(dist)
    return index_nearest
    

URL1='current_08hind_hourly.nc'
ds = Dataset(URL1,'r').variables   # netCDF4 version
URL='gom6-grid.nc'
ds1 = Dataset(URL,'r').variables   # netCDF4 version

st_lat=[]
st_lon=[]
latc=np.linspace(41.8,42.0,10)
lonc=np.linspace(-70.5,-70.2,10)
for aa in np.arange(len(lonc)):
    for bb in np.arange(len(latc)):
        st_lat.append(latc[bb])
        st_lon.append(lonc[aa])
        
T='1858-11-17T00:00:00Z'
time=[]
for a in np.arange(len(ds['ocean_time'])-1):
    #print ds['ocean_time'][a+1]-ds['ocean_time'][a]
    drt = datetime.strptime(T,'%Y-%m-%dT%H:%M:%SZ')+timedelta(hours=ds['ocean_time'][a]/float(3600)) 
    time.append(drt)
FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])

days=7

start_time=datetime(2008,8,1,0,0,0,0)
m_ps =dict(lon=[],lat=[],time=[])

end_time=start_time+timedelta(hours=days*24)
index11=np.argmin(abs(np.array(time)-start_time))#index in data
index22=np.argmin(abs(np.array(time)-end_time))

lon_u=np.hstack(ds1['lon_u'][:])
lat_u=np.hstack(ds1['lat_u'][:])
lon_v=np.hstack(ds1['lon_v'][:])
lat_v=np.hstack(ds1['lat_v'][:])
cc=[]
#########################################################
uu=[]
vv=[]
b=0
#v0=np.hstack(ds['v'][index11][-1][:][:])
#u0=np.hstack(ds['u'][index11][-1][:][:])
v0=np.hstack(ds['v'][index11][-1][:][:])
u0=np.hstack(ds['u'][index11][-1][:][:])
"""
for a in np.arange(len(u0)):
    if u0[a]>100:#this means the point is at land, not in ocean
        u0[a]=0
for a in np.arange(len(v0)):
    if v0[a]>100:
        v0[a]=0
"""        

##########################################################33
#lat_v=la(np.hstack(ds1['lon_v'][:]),np.hstack(ds1['lat_v'][:]))

plt.figure()
plt.scatter(st_lon,st_lat,color='green',zorder=1)
plt.scatter(st_lon[0],st_lat[0],color='green',label='start',zorder=1)

roms=[]
nodes = dict(lon=[st_lon[0]], lat=[st_lat[0]],time=[time[index11]])
for a in np.arange(len(st_lon)):
    print 'a:',a
    #v0=np.hstack(ds['v'][index11][-1][:][:])
    #u0=np.hstack(ds['u'][index11][-1][:][:])
    index1=nearlonlat(lon_u,lat_u,st_lon[a],st_lat[a])#index in grid
    print 'index1:',index1
    index2=nearlonlat(lon_v,lat_v,st_lon[a],st_lat[a])
    print 'index2:',index2
    
    u, v = rot2d(u0[index1], v0[index2], ds1['angle'][0][0])
    
    #print 'u:',u
    #print 'v:',v
    if v>100:
        v=0
        u=0
        index_nearest=find_nearest_coast(st_lon[a],st_lat[a],CL)#get the index of nearest point on coastline
        lon=CL[index_nearest][0]
        lat=CL[index_nearest][1]
        print '111111111111'
        #continue
        #break
    else:
        dx = 60*60*u; dy = 60*60*v#get the distance after one hour  
        lon = st_lon[a] + dx/(111111*np.cos(st_lat[a]*np.pi/180))  
        lat = st_lat[a] + dy/111111
    nodes['lon'].append(lon)    
    nodes['lat'].append(lat)
    nodes['time'].append(time[index11])

    for c in np.arange(1,index22-index11):
        
        #v1=np.hstack(ds['v'][c+index11][-1][:][:])
        #u1=np.hstack(ds['u'][c+index11][-1][:][:])
        v1=np.hstack(ds['v'][c+index11][-1][:][:])
        u1=np.hstack(ds['u'][c+index11][-1][:][:])
        index_lon=nearlonlat(lon_u,lat_u,lon,lat)
        index_lat=nearlonlat(lon_v,lat_v,lon,lat)
        '''
        if mask_u[index1]==1 or mask_v==1:
            break
        '''
        u1_t, v1_t = rot2d(u1[index_lon], v1[index_lat], ds1['angle'][0][0])
        if v1_t>100:
            v1_t=0
            u1_t=0
            index_nearest=find_nearest_coast(st_lon[a],st_lat[a],CL)#get the index of nearest point on coastline
            print 'index_nearest:',index_nearest
            lon=CL[index_nearest][0]
            lat=CL[index_nearest][1]
            print '22222222222222'
            break
        else:
            dx1 = 60*60*u1_t; dy1 = 60*60*v1_t
            lon = lon + dx1/(111111*np.cos(lat*np.pi/180))
            lat = lat + dy1/111111
        nodes['lon'].append(lon)
        nodes['lat'].append(lat)
        nodes['time'].append(time[c+index11])
        roms.append(nodes)
    plt.scatter(nodes['lon'][-1],nodes['lat'][-1],color='red',zorder=1)
    #plt.plot([st_lon[a],nodes['lon'][-1]],[st_lat[a],nodes['lat'][-1]],'y-')
    plt.plot(nodes['lon'][0:],nodes['lat'][0:],'y-',zorder=0)
    cc.append(nodes)

plt.scatter(nodes['lon'][-1],nodes['lat'][-1],color='red',label='end',zorder=1)
#np.save('ROMS'+'_'+str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+days+'days',roms)
np.save('ROMS_2008-8-1_depth=0',roms)
plt.legend(loc='lower center') 
plt.plot(CL['lon'],CL['lat'],'b-',linewidth=0.5) 

plt.xlim([-70.7,-69.9])
plt.ylim([41.5,42.1])
#plt.title(str(start_time)+' '+days+'days'+' surface particle trajectories',fontsize=14)
plt.title('2008-8-1 surface particle trajectories',fontsize=14)
#plt.savefig('ROMS'+'_'+str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+days+'days',dpi=200)
plt.savefig('ROMS_2008-8-1_depth=0',dpi=200)