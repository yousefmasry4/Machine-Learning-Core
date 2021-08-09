# Face Server


## Run this project using docker-compose

Inside the root project you can run

```shell
sudo docker-compose build
```

and then run the following to start the container and expose the API:

```shell
sudo docker-compose up
```

## Api route

**http://0.0.0.0:5000/maskImage** _[Post]_

## Api documentation

### Parameters:

-image

#### The following you can ignore and only send what you need

- face_oval
- left_eye
- right_eye
- left_eye_brow 
- right_eye_brow 

**For example if you need only face_oval and left_eye :**

_C# example we send only face_oval and left_eye and ignore others_

```shell
var request = http.MultipartRequest('POST', Uri.parse('http://0.0.0.0:5000/maskImage'));
request.fields.addAll({
  'face_oval': '',
  'left_eye': '',
});
request.files.add(await http.MultipartFile.fromPath('image', 'path to img'));

http.StreamedResponse response = await request.send();

if (response.statusCode == 200) {
  print(await response.stream.bytesToString());
}
else {
  print(response.reasonPhrase);
}
```
### response:

_Note that i removed part of this response detected in pointed part_

```json
{
    "Points": {
        "face_oval": [
            [
                368.0,
                153.0
            ],
            .
            .
            .
            [
                368.0,
                153.0
            ]
        ],
        "left_eye": [
            [
                575.0,
                223.0
            ],
            .
            .
            .
            [
                467.0,
                236.0
            ]
        ]
    }
}
```
- each array element contain x,y pixel position
_Note that we send image in Base64_
