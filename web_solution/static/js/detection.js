var API_URL = "http://0.0.0.0:8000/";

$(document).ready(function () {
    $('#customFile').change(function () {
        let fileName = $(this).val().split('\\').pop();
        document.getElementById('UploadLabel').innerHTML = fileName
    });
});
    
function upload()
{
    
    console.log(document.getElementById('customFile'))
    var form_data = new FormData();
    var ins = document.getElementById('customFile').files.length;
    console.log("ins is : ", ins, " files : ", document.getElementById('customFile').files[0]);
    if (ins == 0) {
        $('#msg').html('<span style="color:red">Select at least one file</span>');
        return;
    }

    //				for (var x = 0; x < ins; x++) {
    form_data.append("files[]", document.getElementById('customFile').files[0]);
    //				}
    console.log("Form Data : ", form_data)
    $.ajax({
        url: '/', // point to server-side URL
        dataType: 'json', // what to expect back from server
        cache: false,
        contentType: false,
        processData: false,
        data: form_data,
        type: 'post',
        success: function (response) { // display success response
            axios.post(API_URL + '/', {
                filelocation: response.filepath
            })
            if (response.filename)
            {
                for (var x = 0; x < 100000; x++)
                {
                    console.log(document.getElementById('customFile'))
                }
                window.location = "/process/"+response.filename;
            }
        }
    
    });
    
    
        
}