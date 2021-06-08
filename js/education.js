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
    "major": "Computer Engineering",
    "minor": "Mathematics",
    "thumbnail": "about/uiuc_seal.png"
  },
];

$(document).ready(function() {
  $.each(schools, function(school_index, school) {
    $("#schools").append(
      $("<div/>", {"class": "col-12 col-sm-6 col-md-4"}).append(
        $("<div/>", {"class": "col-4 offset-4 col-md-6 offset-md-3"}).append(
          $("<img/>", {"class": "img-fluid my-2 rounded-circle", "src": school.thumbnail}),
        ),
        $("<p/>", {"class": "text-center text-nowrap"}).append(
          school.name,
          "<br>",
          $("<small/>").append(
            school.degree,
            "<br>"
          ).append(
            function() {
              if (school.minor) {
                return "Major in "+school.major+"<br>Minor in "+school.minor;
              } else {
                return school.major;
              }
            }
          )
        )
      )
    );
  });
});