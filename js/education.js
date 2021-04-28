const schools = [
  {
    "name": "Stanford University",
    "degree": "Doctor of Philosophy",
    "major": "Computer Science",
    "thumbnail": "about/stanford_seal.png"
  },

  {
    "name": "Stanford University",
    "degree": "Master of Science",
    "major": "Computer Science",
    "thumbnail": "about/stanford_seal.png"
  },

  {
    "name": "University of Illinois Urbana-Champaign",
    "degree": "Bachelor of Science",
    "major": "Electrical and Computer Engineering",
    "thumbnail": "about/uiuc_seal.png"
  },
]

$(document).ready(function() {
  $.each(schools, function(school_index, school) {
    $("#schools").append(
      $("<div/>", {"class": "col-sm-4 col-xs-6 text-center"}).append(
        $("<img/>", {"class": "my-2 rounded-circle", "src": school.thumbnail, "width": 150, "height": 150}),
        $("<p/>").append(
          school.name,
          "<br>",
          $("<small/>").append(
            school.degree,
            "<br>",
            school.major
          )
        )
      )
    );
  });
});