# This code is to pick a song and have the code recommend a song
# by another artist of your choice that is closest to the original
# song that you picked based on energy and valence using the Spotify API

#########################################
##### Matching Songs Across Artists #####
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn import linear_model, metrics
from numpy.linalg import norm



cid = 
secret = 
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# Billy Joel and Beatles Example
# Get Artist Uri's
name = '{Beatles}' #chosen artist
result = sp.search(name) #search query
result['tracks']['items'][0]['artists']

######################################

###### Pick Artists ######

# base is the artist of the song you want to base the recommendation
base = 'Billy Joel'

# recommended is the artist to want to pull a new song from
recommended = "Beatles"

###### Make song recommendation ######

# song_choice is the song you want the recommendation to sound like
song_choice = 'Piano Man'

######################################

artists = [base, recommended]

artist_df = {}

for a in artists:
    name = a #chosen artist
    result = sp.search(name) #search query
    result['tracks']['items'][0]['artists']


    #Extract Artist's uri
    artist_uri = result['tracks']['items'][0]['artists'][0]['uri']
    #Pull all of the artist's albums
    sp_albums = sp.artist_albums(artist_uri, album_type='album')
    #Store artist's albums' names' and uris in separate lists
    album_names = []
    album_uris = []
    for i in range(len(sp_albums['items'])):
        album_names.append(sp_albums['items'][i]['name'])
        album_uris.append(sp_albums['items'][i]['uri'])
        
    album_names
    album_uris
    zipped = list(zip(album_names, album_uris))
    album_names_uris = pd.DataFrame(zipped, columns=['name', 'uri'])
    #Keep names and uris in same order to keep track of duplicate albums
    
    spotify_albums = {}
    
    def albumSongs(uri):
        album = uri #assign album uri to a_name
    
    for i in album_uris:
        spotify_albums[i] = {} #Creates dictionary for that specific album
    #Create keys-values of empty lists inside nested dictionary for album
        spotify_albums[i]['album'] = [] #create empty list
        spotify_albums[i]['track_number'] = []
        spotify_albums[i]['id'] = []
        spotify_albums[i]['name'] = []
        spotify_albums[i]['uri'] = []
        
        spotify_albums[i]['album'].append(i) #append album name tracked via album_count
    
      
        tracks = sp.album_tracks(i) #pull data on album tracks
        for n in range(len(tracks['items'])): #for each song track
            spotify_albums[i]['track_number'].append(tracks['items'][n]['track_number'])
            spotify_albums[i]['id'].append(tracks['items'][n]['id'])
            spotify_albums[i]['name'].append(tracks['items'][n]['name'])
            spotify_albums[i]['uri'].append(tracks['items'][n]['uri'])
    
    album_count = 0
    for i in album_uris: #each album
        albumSongs(i)
        print("Album " + str(album_names[album_count]) + " songs has been added to spotify_albums dictionary")
        album_count+=1 #Updates album count once all tracks have been added
    
    def audio_features(album):
        #Add new key-values to store audio features
        spotify_albums[album]['acousticness'] = []
        spotify_albums[album]['danceability'] = []
        spotify_albums[album]['energy'] = []
        spotify_albums[album]['instrumentalness'] = []
        spotify_albums[album]['liveness'] = []
        spotify_albums[album]['loudness'] = []
        spotify_albums[album]['speechiness'] = []
        spotify_albums[album]['tempo'] = []
        spotify_albums[album]['valence'] = []
        spotify_albums[album]['popularity'] = []
        #create a track counter
        track_count = 0
        for track in spotify_albums[album]['uri']:
            #pull audio features per track
            features = sp.audio_features(track)
            
            #Append to relevant key-value
            spotify_albums[album]['acousticness'].append(features[0]['acousticness'])
            spotify_albums[album]['danceability'].append(features[0]['danceability'])
            spotify_albums[album]['energy'].append(features[0]['energy'])
            spotify_albums[album]['instrumentalness'].append(features[0]['instrumentalness'])
            spotify_albums[album]['liveness'].append(features[0]['liveness'])
            spotify_albums[album]['loudness'].append(features[0]['loudness'])
            spotify_albums[album]['speechiness'].append(features[0]['speechiness'])
            spotify_albums[album]['tempo'].append(features[0]['tempo'])
            spotify_albums[album]['valence'].append(features[0]['valence'])
            #popularity is stored elsewhere
            pop = sp.track(track)
            spotify_albums[album]['popularity'].append(pop['popularity'])
            track_count+=1
    
    import time
    sleep_min = 2
    sleep_max = 5
    start_time = time.time()
    request_count = 0
    for i in spotify_albums:
        audio_features(i)
        request_count+=1
        if request_count % 5 == 0:
            print(str(request_count) + " playlists completed")
            time.sleep(np.random.uniform(sleep_min, sleep_max))
            print('Loop #: {}'.format(request_count))
            print('Elapsed Time: {} seconds'.format(time.time() - start_time))        
    
    dic_df = {}
    dic_df['album'] = []
    dic_df['track_number'] = []
    dic_df['id'] = []
    dic_df['name'] = []
    dic_df['uri'] = []
    dic_df['acousticness'] = []
    dic_df['danceability'] = []
    dic_df['energy'] = []
    dic_df['instrumentalness'] = []
    dic_df['liveness'] = []
    dic_df['loudness'] = []
    dic_df['speechiness'] = []
    dic_df['tempo'] = []
    dic_df['valence'] = []
    dic_df['popularity'] = []
    for album in spotify_albums: 
        for feature in spotify_albums[album]:
            dic_df[feature].extend(spotify_albums[album][feature])
    len(dic_df['album'])
    dic_df.pop('album')
    
    df = pd.DataFrame.from_dict(dic_df)
    df.head()
    
    df = df.sort_values('popularity', ascending=False).drop_duplicates('name').sort_index()
    
    # Mood-Based Recommendation System (Energy and Valence)
    df['mood_vec'] = df[['valence', 'energy']].values.tolist()
    
    df = df.set_index('id')

    artist_df[a] = df


def distance(p1, p2):
    distance_x = p2[0]-p1[0]
    distance_y = p2[1]-p1[1]
    distance_vec = [distance_x, distance_y]
    norm = (distance_vec[0]**2 + distance_vec[1]**2)**(1/2)
    return norm

def recommend(track_id, base_df, recommend_df, n_recs = 5):
    
    energy = base_df._get_value(track_id, 'energy')
    valence = base_df._get_value(track_id, 'valence')
    track_moodvec = np.array([[energy], [valence]])
    
    # Compute distances to all reference tracks
    recommend_df['distances'] = recommend_df['mood_vec'].apply(lambda x: norm(track_moodvec-np.array(x)))
    # Sort distances from lowest to highest
    recommend_df_sorted = recommend_df.sort_values(by = 'distances', ascending = True)
    # If the input track is in the reference set, it will have a distance of 0, but should not be recommendet
    recommend_df_sorted = recommend_df_sorted[recommend_df_sorted.index != track_id]

    # Return n recommendations
    return recommend_df_sorted['name'].iloc[:n_recs]


song_choice_uri = artist_df[base][artist_df[base]['name'] == song_choice].index[0]
recommended_songs = recommend(track_id = song_choice_uri, base_df = artist_df[base], 
          recommend_df = artist_df[recommended], n_recs = 5)

print("Best Match:", recommended_songs.iloc[0])
print("2nd Best Match:", recommended_songs.iloc[1])
print("3rd Best Match:", recommended_songs.iloc[2])


# Compare valence and energy
print('Chosen-Song Energy:', artist_df[base]._get_value(song_choice_uri, 'energy'))
print('Chosen-Song Valence:', artist_df[base]._get_value(song_choice_uri, 'valence'))

print('Best-Match Energy:', artist_df[recommended]._get_value\
      (recommended_songs[recommended_songs == recommended_songs.iloc[0]].index[0], 
       'energy'))
print('Best-Match Valence:', artist_df[recommended]._get_value\
      (recommended_songs[recommended_songs == recommended_songs.iloc[0]].index[0], 
       'valence'))

