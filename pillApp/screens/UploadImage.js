import { StatusBar } from 'expo-status-bar';
import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, View, Button, Image, Alert, TouchableOpacity, Text, ScrollView} from 'react-native'
import * as ImagePicker from 'expo-image-picker';
import { Camera, CameraType, takePictureAsync } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library';
import { Menu, MenuProvider, MenuOptions, MenuOption, MenuTrigger} from 'react-native-popup-menu';

const UploadImage = ({route, navigation}) => {
  const pressHandler1 = () => {
    // navigation.navigate('Result');
    navigation.navigate('Result',{IMG:image1, IMG2:image2});
  }
  const pressCamera = () => {
    navigation.navigate('CameraPhoto');
  }
 

  // Image Upload
  const [hasPermission, setHasPermission] = useState(null);
  const [hasCamPermission, setHasCamPermission] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [type, setType] = useState(CameraType.back);
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [hasimage, setHasImage] = useState(false);
  const [whatImage, setWhatImage] = useState(null);
  // camera ref to access camera
  const cameraRef = useRef(null);


  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasCamPermission(status === 'granted');
    })();
  }, []);

  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  if (hasCamPermission === null) {
    return <View />;
  }
  if (hasCamPermission === false) {
    return <Text>No access to camera</Text>;
  }

  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to MediaLibrary</Text>;
  }

  // useEffect(() => {
  //   fetch("http://172.30.1.60:5000/image", {
  //     method : 'GET'
  //   })
  //   .then(response => response.json())
  //   .then(data => {
  //       console.log(data)
  //       // setImage(data.uri);
  //     })
  // },[])

  const pickImage1 = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });
    // const formData = new FormData();
    // formData.append('imageInfo', {
    //   uri: result.assets[0].uri,
    //   type: result.assets[0].type,
    //   name: result.assets[0].fileName,
    // })
    
    // console.log("image pick in Library: ", result.assets);

    if (!result.canceled) {
      setImage1(result.assets[0].uri)   
      setHasImage(true)
    //   let res = await fetch(
    //     'http://172.30.1.60:5000/image', {
    //       method: 'POST',
    //       headers:{
    //         Accept: 'application/json',
    //         'Content-Type':'application/json', 
    //       },
    //       body: JSON.stringify({
    //         uri: result.assets[0].uri,
    //         type: result.assets[0].type,
    //         name: result.assets[0].fileName,
    //       }),
    //     }
    //   )
    //   .then((response) => response.json())
    //   .then((responseData) => {
    //     console.log(responseData, 'responseJson')
    //     setImage(responseData.uri)
    //   })
    }
  };
  const pickImage2 = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });
    if (!result.canceled) {
      setImage2(result.assets[0].uri)
      setHasImage(true)
    }
    
  };
  const takePhoto = async() => {
    if (cameraRef) {
      console.log("in take picture");
      try {
        let photo = await cameraRef.current.takePictureAsync({
          forceUpOrientation: true,
          aspect: [4,3],
          quality: 1,
        });
        const asset = await MediaLibrary.createAssetAsync(photo.uri);
        return photo;
      } catch (e) {
        console.log(e);
      }
    }
  };

  return (
    <View style={{flex:1}}>
      {showCamera ? (<Camera style={{flex: 1,}} type={type} ref={cameraRef}>
        <View style={{flex:1, alignItems:'center', justifyContent:'center'}}>
          <Image style={{
            width: 300, 
            height: 300,
            borderRadius: 30,
            marginTop:350,
          }} source = {
              require('../assets/Dotted_Box2.png')
            }/>
        </View>
        <View style={cam_styles.buttonContainer}>
        <View style = {
            { 
              "flexDirection": "row",
              "alignSelf": "flex-end",
              "alignItems": "center",
              "paddingTop": 5,
              "width": 390,
              "height": 100,
              "backgroundColor": "rgba(255, 255, 255, 255)"
            }
          } >
          <TouchableOpacity
            style={cam_styles.button}
            onPress={() => {
              setType(type === CameraType.back ? CameraType.front : CameraType.back);
            }}>
            
            <Image style = {
            {
              "width": 24,
              "height": 24,
              "marginBottom":60,
              "marginLeft":50
            }
            }
            source = {
              require('../assets/circle.png')
            }/>
          </TouchableOpacity>
          <TouchableOpacity
            style={cam_styles.button}
            onPress={async() => {
              const r = await takePhoto();
              if (whatImage == "Front") {
                setImage1(r?.uri)
                setHasImage(true)
                setShowCamera(false)
              }
              else{
                setImage2(r?.uri)
                setHasImage(true)
                setShowCamera(false)
              }
            }}
            >
            <View style = {
              {
                "alignItems": "center",
                "paddingTop": 9,
                "marginBottom":50,
                "width": 100,
                "height": 42,
                "borderRadius": 21,
                "backgroundColor": "rgba(67, 175, 244, 255)"
              }
            } >
            <Image style = {
            {
              "width": 24,
              "height": 24,
              "marginBottom":60
            }
            }
            source = {
              require('../assets/camera.png')
            }/>
            </View>
          </TouchableOpacity>
          <TouchableOpacity
            style={cam_styles.button}
            onPress={async() => {
              setShowCamera(false);
            }}>
            <Image style = {
            {
              "width": 24,
              "height": 24,
              "marginBottom":60,
              "marginRight":50
            }
            }
            source = {
              require('../assets/back.png')
            }/>
          </TouchableOpacity>
        </View>
        </View>
      </Camera>
      ) : (
      <View style={styles.container}>
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center'}}>
            <MenuProvider style={styles.menu_style}>
              <Image style = {
              {
                // "marginStart": 20,
                "width": 180,
                "height": 101.25,
                "marginBottom": 20
              }
              }
              source = {
                  require('../assets/pillm.png')
              }/>
              <Menu style={{marginBottom: 20,}}>
                <MenuTrigger style={{
                    backgroundColor:"rgba(67, 175, 244, 255)",
                    width:212,
                    height:42,
                    borderRadius: 21,
                    justifyContent:'center',
                    margin:0,
                    alignItems:'center',
                  }}>
                  <Text style={{color: 'white'}}>
                    TAKE PICTURES
                  </Text>
                </MenuTrigger>
                <MenuOptions
                  optionsContainerStyle={{
                    borderRadius: 10,
                    marginTop:-30,
                    backgroundColor:'#F3F5F7',
                    marginLeft:15,
                  }}
                >
                  <MenuOption 
                    onSelect={() => {setShowCamera(true); setWhatImage('Front')}} 
                    text="Front Image" 
                    customStyles={{
                      optionWrapper: {height: 30, marginLeft: 5, marginRight: 5, marginTop: 5, marginBottom: 1},
                      optionText: {fontSize: 16},
                    }}
                  />
                  <View style={styles.divider} />
                  <MenuOption
                    onSelect={() => {setShowCamera(true); setWhatImage('Back')}}
                    text="Back Image"
                    customStyles={{
                      optionWrapper: {height: 30, marginLeft: 5, marginRight: 5, marginTop: 5, marginBottom: 1},
                      optionText: {fontSize: 16},
                    }}
                  />
                </MenuOptions>
              </Menu>

              <Menu>
                <MenuTrigger style={{
                    backgroundColor:"rgba(67, 175, 244, 255)",
                    width:212,
                    height:42,
                    borderRadius: 21,
                    justifyContent:'center',
                    margin:0,
                    alignItems:'center',
                  }}>
                  <Text style={{color: 'white'}}>
                    SELECT IMAGES
                  </Text>
                </MenuTrigger>
                <MenuOptions
                  optionsContainerStyle={{
                    borderRadius: 10,
                    marginTop:-30,
                    backgroundColor:'#F3F5F7',
                    marginLeft:15,
                  }}
                >
                  <MenuOption 
                    onSelect={pickImage1}
                    text="Front Image" 
                    customStyles={{
                      optionWrapper: {height: 30, marginLeft: 5, marginRight: 5, marginTop: 5, marginBottom: 1},
                      optionText: {fontSize: 16},
                    }}
                  />
                  <View style={styles.divider} />
                  <MenuOption
                    onSelect={pickImage2}
                    text="Back Image"
                    customStyles={{
                      optionWrapper: {height: 30, marginLeft: 5, marginRight: 5, marginTop: 5, marginBottom: 1},
                      optionText: {fontSize: 16},
                    }}
                  />
                </MenuOptions>
              </Menu>
            </MenuProvider>
        </View>
        {hasimage && <ScrollView style = {{height: 10, marginLeft: 40, marginRight: 40}} horizontal={true}>
          {image1 && <Image source={{ uri: image1 }} style={styles.img_style} resizeMethod='contain'/>}
          {image2 && <Image source={{ uri: image2 }} style={styles.img_style} onLoadEnd={() => alert('Swipe to preview image')} resizeMethod='contain'/>}
        </ScrollView>}
        {hasimage &&<View style={{marginBottom:30, flex: 0, flexDirection: 'row', alignItems: 'center', justifyContent: 'center'}}>
        <TouchableOpacity onPress={() => {setImage1(null); setImage2(null); setHasImage(false); setWhatImage(null)}}>
          <View style = {
            {
              "alignItems": "center",
              "justifyContent":"center",
              "margin":8,
              "width": 42,
              "height": 42,
              "borderRadius": 99,
              "backgroundColor": "rgba(154, 167, 191, 255)",
            }
          } >
          <Image style = {
            {
              "width": 20,
              "height": 20,
              // "marginBottom":60,
              // "marginLeft":50
            }
            }
            source = {
              require('../assets/clear.png')
            }/>
          {/* <Text style = {styles.button_style} >RESET</Text> */}
          </View>
        </TouchableOpacity>
        <TouchableOpacity  onPress={() => {
          if (image1 && image2){
            pressHandler1()
          } else{
            alert("Two Image Upload!!")
          }
        }}>
          <View style = {
            {
              "alignItems": "center",
              "justifyContent":"center",
              "margin":8,
              "width": 170,
              "height": 42,
              "borderRadius": 99,
              "backgroundColor": "rgba(70, 83, 107, 255)",
            }
          } >
          
          <Text style = {styles.button_style} >SUBMIT</Text>
          </View>
        </TouchableOpacity>
        </View>
        }
        
      </View>
      )}
      
    </View>
    
  );
}


const cam_styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    // alignItems: 'center',
    // justifyContent: 'center',
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    margin: 0,
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  }
});

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 80,
  },
  img_style: { 
    // marginTop: 20,
    width: 300, 
    height: 300,
    borderRadius: 30,
    marginLeft: 5,
  },
  button_style: {
    // fontFamily: "NanumSquareOTF",
    fontSize: 16,
    // fontWeight: 'bold',
    textAlign: "center",
    color: "rgba(255, 255, 255, 255)",
  },
  menu_style:{
    flex: 1,
    backgroundColor: "#fff",
    justifyContent: "center",
    alignItems: "center",
    // marginVertical: 100,
    marginHorizontal: 100,
    // flexDirection: "column",
    // borderRadius: 30,
    paddingBottom: 30,
  },
  divider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "#7F8487",
  },
});

export default UploadImage;