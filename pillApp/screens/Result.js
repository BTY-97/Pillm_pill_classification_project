import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, FlatList, Pressable, Image, SafeAreaView } from 'react-native';
import axios from 'axios';


export default function Result({route, navigation}) {
    const DetailPage = (props) => {
        navigation.navigate('Detail',{'pillItem':props});
    };
    const image1 = route.params.IMG;
    const image2 = route.params.IMG2;

    const [isLoading, setLoading] = useState(true);
    const [data, setData] = useState(null);

    useEffect(() => {
    
      const localUri = image1.toString();
      const filename = localUri.split('/').pop();
      const match = /\.(\w+)$/.exec(filename ?? '');
      const type = match ? `image/${match[1]}` : `image`;
      const formData = new FormData();
      formData.append('images', { uri: localUri, name: filename, type });
      console.log(image2);
      const localUriB = image2.toString();
      const filenameB = localUriB.split('/').pop();
      const matchB = /\.(\w+)$/.exec(filenameB ?? '');
      const typeB = matchB ? `image/${matchB[1]}` : `image`;
      // // const formData = new FormData();
      formData.append('images', { uri: localUriB, name: filenameB, typeB });
      
      axios({
        method: 'post', 
        url: 'http://minina.iptime.org:9599/prediction/',
        headers: {
          'content-type': 'multipart/form-data',
        },
        data: formData
      })
      .then(response => {
        setData(response.data);
        setLoading(false);
      });
      
    }, []);

    // useEffect(() => {
    //   fetch('http://172.30.1.51:8000/',{ // fast api 주소 따라서 변경해야함.
    //     Accept: 'application/json',
    //   })
    //   .then((response) => response.json())
    //   .then((json) => setData(json))
    //   .catch((error) => console.error(error))
    //   .finally(() => setLoading(false));
    // },[])
    
    const ListItem = ({item}) => {
      return (
        <Pressable style={styles.listItem} onPress={() => DetailPage(item)}>
          <View style={styles.pillContainer}>
            <Image
              source={{uri: item.IMAGE}}
              style={styles.listImage}
            />
            <View style={styles.pillContent}>
              <Text style={{fontSize: 20, fontWeight: '700', marginBottom: 5, color:'#46536b'}}>{String(item.NAME).split('(')[0]}</Text>
              <Text style={{fontSize: 16, fontWeight: '300', color:'#46536b'}}>{item.MY} / {item.COLOR}</Text>
            </View>
          </View>
          
        </Pressable>
      );
    };
    const ListHeaderComponent = (props) => {
      return (
        <View
          style={{
            height: 40,
            marginTop: 20,
            width:999
          }}
        >
          <Text style={{marginLeft: 20,fontSize: 18, color:'#46536b'}}>총 <Text style={{fontWeight: 'bold', fontSize: '30', color:'#43aff4'}}>{data.length}</Text>건의 검색결과가 있습니다.</Text>
        </View>
          
      )
    };
    console.log(data);
    return (
      <View style = {styles.container}>
      <SafeAreaView>
        {isLoading ? <Text>Loading...</Text> :
        (<FlatList
          data={data}
          ListHeaderComponent={ListHeaderComponent}
          renderItem={ListItem}
          keyExtractor={item => item.PK}
        />)}
      </SafeAreaView>
      </View>
    );
  }
  
  const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
    },
    listImage: {
      width: 120,
      // height: '100%',
      height: 80,
      borderRadius: 10,
      alignItems: 'center',
      justifyContent: 'center',
    },
    listItem:{
      height : '100%',
      marginBottom: 5,
      paddingHorizontal: 15,
      flex: 1,
    },
    pillContainer: {
      flexDirection: 'row',
      alignContent: 'center',
      padding: 10,
      borderRadius: 20,
      margin:1,
      backgroundColor: '#fff',
      ...Platform.select({
        ios: {
          shadowColor: "#3b82d4",
          shadowOpacity: 0.25,
          shadowRadius: 2,
          shadowOffset: {
            height: 1,
            width: 0,
          },
        },
        android: {
          elevation: 1,
        },
      })
      
      // flex: 1,
    },
    pillContent:{
      marginLeft: 20,
      // marginRight: 20,
      justifyContent:'center',
      overflow: 'visible',
      flex: 1,
    },
  });