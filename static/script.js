function getResponse() {
    let userText = $("#textInput").val();
    
    // Check if input is empty
    if (userText.trim() == "") {
        return;
    }

    // 1. Add user message to the chatbox
    let userHtml = '<p class="userText"><span>' + userText + '</span></p><div style="clear: both;"></div>';
    $("#chatbox").append(userHtml);
    $("#textInput").val(""); // Clear input
    
    // 2. Scroll to bottom
    $("#chatbox").scrollTop($("#chatbox")[0].scrollHeight);

    // 3. Ask the Python server for a response
    $.get("/get", { msg: userText }).done(function(data) {
        var botHtml = '<p class="botText"><span>' + data + '</span></p><div style="clear: both;"></div>';
        $("#chatbox").append(botHtml);
        $("#chatbox").scrollTop($("#chatbox")[0].scrollHeight);
    });
}

// Trigger button click on "Enter" key
$("#textInput").keypress(function(e) {
    if(e.which == 13) {
        getResponse();
    }
});

// Trigger on click
$("#buttonInput").click(function() {
    getResponse();
});
