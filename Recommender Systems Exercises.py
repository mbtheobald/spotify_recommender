# Recommender Systems Exercise (Spotify Recommendations)

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn import linear_model, metrics
from numpy.linalg import norm


cid = 'a71d12b3506b4afd9e4ea3ebb1537802'
secret = '9cf79afdf35249ec93f75b6616b26502'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# Beatles Example
name = '{Beatles}' #chosen artist
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

print(len(df))
final_df = df.sort_values('popularity', ascending=False).drop_duplicates('name').sort_index()
print(len(final_df))

final_df.to_csv('Beatles Songs.csv')

# Linear Regression
y = final_df['popularity']
X = final_df.iloc[:,4:-1]

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=123)
  
# create linear regression object
reg = linear_model.LinearRegression()
  
# train the model using the training sets
train_reg = reg.fit(X_train, y_train)
  
# regression coefficients
print('Coefficients: ', reg.coef_)
  
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

feature_coefficients = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficient'])  
feature_coefficients

y_pred = reg.predict(X_test)

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
comparison

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

# Mood-Based Recommendation System (Energy and Valence)
final_df['mood_vec'] = final_df[['valence', 'energy']].values.tolist()

final_df = final_df.set_index('id')

def distance(p1, p2):
    distance_x = p2[0]-p1[0]
    distance_y = p2[1]-p1[1]
    distance_vec = [distance_x, distance_y]
    norm = (distance_vec[0]**2 + distance_vec[1]**2)**(1/2)
    return norm

def recommend(track_id, ref_df, sp, n_recs = 5):
    
    energy = ref_df._get_value(track_id, 'energy')
    valence = ref_df._get_value(track_id, 'valence')
    track_moodvec = np.array([[energy], [valence]])
    
    # Compute distances to all reference tracks
    ref_df['distances'] = ref_df['mood_vec'].apply(lambda x: norm(track_moodvec-np.array(x)))
    # Sort distances from lowest to highest
    ref_df_sorted = ref_df.sort_values(by = 'distances', ascending = True)
    # If the input track is in the reference set, it will have a distance of 0, but should not be recommendet
    ref_df_sorted = ref_df_sorted[ref_df_sorted.index != track_id]

    # Return n recommendations
    return ref_df_sorted['name'].iloc[:n_recs]

day_tripper = '29b2b96jozyD9GPCkOrVLs'
recommend(track_id = day_tripper, ref_df = final_df, sp = sp, n_recs = 5)

print('Chosen-Song Energy:', final_df._get_value(day_tripper, 'energy'))
print('Chosen-Song Valence:', final_df._get_value(day_tripper, 'valence'))

print('Best-Match Energy:', final_df._get_value('52omU1KgD4yyRkXx2gDgZj', 'energy'))
print('Best-Match Valence:', final_df._get_value('52omU1KgD4yyRkXx2gDgZj', 'valence'))



song_names = final_df['name'].tolist()

indices = [i for i, x in enumerate(song_names.split()) if x == 'Tripper']

if __name__ == '__main__':
    song = 'Day Tripper'
    indices = [i for i, j in enumerate(final_df['name']) if j == song]
    print(f'Song {song} is found at index {indices}')
 
 

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



cid = 'a71d12b3506b4afd9e4ea3ebb1537802'
secret = '9cf79afdf35249ec93f75b6616b26502'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# Billy Joel and Beatles Example
# Get Artist Uri's
name = '{Lynyrd Skynyrd}' #chosen artist
result = sp.search(name) #search query
result['tracks']['items'][0]['artists']


name = '{Beatles}' #chosen artist
result = sp.search(name) #search query
result['tracks']['items'][0]['artists']

######################################

###### Pick Artists ######

base = 'Lynyrd Skynyrd'
recommended = "Beatles"

###### Make song recommendation ######

song_choice = 'Free Bird'

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






###############################
##### Get Larger Datasets #####
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn import linear_model, metrics
from numpy.linalg import norm
from spotipy.oauth2 import SpotifyOAuth

cid = 'a71d12b3506b4afd9e4ea3ebb1537802'
secret = '9cf79afdf35249ec93f75b6616b26502'
scopes = ["user-follow-read", 'ugc-image-upload', 'user-read-playback-state',
          'user-modify-playback-state', 'user-read-currently-playing', 'user-read-private',
          'user-read-email', 'user-follow-modify', 'user-follow-read', 'user-library-modify',
          'user-library-read', 'streaming', 'app-remote-control', 'user-read-playback-position',
          'user-top-read', 'user-read-recently-played', 'playlist-modify-private', 'playlist-read-collaborative',
          'playlist-read-private', 'playlist-modify-public']

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid,
                                               client_secret=secret,
                                               redirect_uri='http://localhost:1414/callback',
                                               scope=scopes))

# Get my account summary
user_data = sp.current_user()
print('My data:')
print('Name:', user_data['display_name'])
print('Followers:', user_data['followers']['total'])
print('Link:', user_data['external_urls']['spotify'])
print('Account:', user_data['product'])

# Get followed artists
print('Followed Artists')
artists = sp.current_user_followed_artists()['artists']
while artists:
    for i, artist in enumerate(artists['items']):
        print(f'{i} - {artist["name"]}')
        print(f'\tFollowers: {artist["followers"]["total"]}')
        print(f'\tPopularity: {artist["popularity"]}')
        print(f'\tGenres: {artist["genres"]}')

    if artists['next']:
        artists = sp.next(artists)
    else:
        artists = None

# Get my playlists
print('Playlists:')
playlists = sp.current_user_playlists()
while playlists:
    for i, playlist in enumerate(playlists['items']):
        print(f'{i} - {playlist["name"]}')
        print(f'\tNumber of tracks: {playlist["tracks"]["total"]}')
        print(f'\tUrl: {playlist["external_urls"]}')
        print(f'\tDescription: {playlist["description"]}')
    if playlists['next']:
        playlists = sp.next(playlists)
    else:
        playlists = None

# Get information in CSC Break Playlist specifically
# CSC Breaks URI: 6OAzbgSTOohe6tvHShT7VT?si=6b3fec71d5d94b3a
playlist = sp.playlist(playlist_id='6OAzbgSTOohe6tvHShT7VT')
print('Playlist name:', playlist['name'])
playlist_tracks = playlist['tracks']
while playlist_tracks:
    for i, track in enumerate(playlist_tracks['items']):
        print(f'{i} - {track["track"]["name"]}')
        print(f'\tAdded at: {track["added_at"]}')
        # from artists key you can access to artists details
        print(f'\tArtist: {track["track"]["artists"][0]["name"]}')
        # from album key you can access to album details
        print(f'\tAlbum: {track["track"]["album"]["name"]}')
        print(f'\tDuration: {round(track["track"]["duration_ms"] / 60000, 2)} minutes')
        print(f'\tPopularity: {track["track"]["popularity"]}')

    if playlist_tracks['next']:
        playlist_tracks = sp.next(playlist_tracks)
    else:
        playlist_tracks = None

























artist_name = []
track_name = []
popularity = []
track_id = []
valence = []
loudness = []
energy = []
tempo = []
instrumentalness = []

for i in range(0,10000,50):
    track_results = sp.search(q='year:2013', type='track', limit=50,offset=i)
    for i, t in enumerate(track_results['tracks']['items']):
        artist_name.append(t['artists'][0]['name'])
        track_name.append(t['name'])
        track_id.append(t['id'])
        popularity.append(t['popularity'])
        
        valence.append(t['valence'])
        loudness.append(t['loudness'])
        energy.append(t['energy'])
        tempo.append(t['tempo'])
        instrumentalness.append(t['instrumentalness'])

        
df = pd.DataFrame({'artist_name' : artist_name, 
                   'track_name' : track_name, 
                   'track_id' : track_id, 
                   'popularity' : popularity,
                   'valence' : valence,
                   'loudness' : loudness,
                   'energy' : energy,
                   'tempo' : tempo,
                   'instrumentalness' : instrumentalness})

print(df.shape)
df.head()     


df.to_csv('2013 Songs.csv')


os.getcwd()

df = pd.read_csv(r'2013 Songs.csv')

df['mood'] = df[['valence', 'energy']].values.tolist()

df_train, df_test = train_test_split(df, test_size=0.25, random_state=123)

# The distance metric used is cosine similarity
