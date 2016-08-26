// -w320 -h240

#version 3.6;

#include "colors.inc"
#include "textures.inc"
#include "shapes.inc"

global_settings {max_trace_level 5 assumed_gamma 1.0}

camera {
	location <-3.000000, 6.000000, -9.000000>
	direction <0, 0,  2.25>
	right x*1.33
	look_at <0,0,0>
}

#declare Dist=80.0;
light_source {< -25, 50, -50> color White
	fade_distance Dist fade_power 2
}
light_source {< 50, 10,  -4> color Gray30
	fade_distance Dist fade_power 2
}
light_source {< 0, 100,  0> color Gray30
	fade_distance Dist fade_power 2
}

sky_sphere {
	pigment {
		gradient y
		color_map {
			[0, 1  color White color White]
		}
	}
}

#declare Xaxis = union{
	cylinder{
		<0,0,0>,<0.8,0,0>,0.05
	}
	cone{
		<0.8,0,0>, 0.1, <1,0,0>, 0
	}
	texture { pigment { color Red } }
}
#declare Yaxis = union{
	cylinder{
		<0,0,0>,<0,0.8,0>,0.05
	}
	cone{
		<0,0.8,0>, 0.1, <0,1,0>, 0
	}
	texture { pigment { color Green } }
}
#declare Zaxis = union{
	cylinder{
	<0,0,0>,<0,0,0.8>,0.05
	}
	cone{
		<0,0,0.8>, 0.1, <0,0,1>, 0
	}
	texture { pigment { color Blue } }
}
#declare Axes = union{
	object { Xaxis }
	object { Yaxis }
	object { Zaxis }
}
#declare Material_ito = texture{ pigment{ rgb <0.783099,0.394383,0.840188> } }
#declare Material_alinp = texture{ pigment{ rgb <0.197551,0.911647,0.798440> } }
#declare Material_gaas = texture{ pigment{ rgb <0.277775,0.768230,0.335223> } }
#declare Material_sio2 = texture{ pigment{ rgb <0.628871,0.477397,0.553970> } }
#declare Material_cyclotrene = texture{ pigment{ rgb <0.952230,0.513401,0.364784> } }
#declare Layer_substrate = union{
	difference{
		intersection{
			plane{ <1.000000,0.000000,0>, 0.500000 }
			plane{ <-1.000000,-0.000000,0>, 0.500000 }
			plane{ <0.000000,1.000000,0>, 0.500000 }
			plane{ <-0.000000,-1.000000,0>, 0.500000 }
			plane{ <1.000000,1.000000,0>, 0.707107 }
			plane{ <-1.000000,-1.000000,0>, 0.707107 }
			plane{ <0,0,-1>, 0 }
			plane{ <0,0,1>, 1000.000000 }
		}
// nshapes = 0
		texture { Material_gaas }
	}
	translate +z*0.000000
}
#declare Layer_nanowire = union{
	difference{
		intersection{
			plane{ <1.000000,0.000000,0>, 0.500000 }
			plane{ <-1.000000,-0.000000,0>, 0.500000 }
			plane{ <0.000000,1.000000,0>, 0.500000 }
			plane{ <-0.000000,-1.000000,0>, 0.500000 }
			plane{ <1.000000,1.000000,0>, 0.707107 }
			plane{ <-1.000000,-1.000000,0>, 0.707107 }
			plane{ <0,0,-1>, 0 }
			plane{ <0,0,1>, 1000.000000 }
		}
// nshapes = 2
cylinder{
	<0,0,0>, <0,0,1000.000000>, 70.000000
	rotate +z*0.000000
	translate +x*0.000000
	translate +y*0.000000
}
		texture { Material_cyclotrene }
	}
	difference{
		intersection{
cylinder{
	<0,0,0>, <0,0,1000.000000>, 70.000000
	rotate +z*0.000000
	translate +x*0.000000
	translate +y*0.000000
}
cylinder{
	<0,0,0>, <0,0,1000.000000>, 50.000000
	rotate +z*0.000000
	translate +x*0.000000
	translate +y*0.000000
}
			plane{ <0,0,-1>, 0 }
			plane{ <0,0,1>, 1000.000000 }
		}
		texture { Material_alinp }
	}
	difference{
		intersection{
cylinder{
	<0,0,0>, <0,0,1000.000000>, 50.000000
	rotate +z*0.000000
	translate +x*0.000000
	translate +y*0.000000
}
			plane{ <0,0,-1>, 0 }
			plane{ <0,0,1>, 1000.000000 }
		}
		texture { Material_gaas }
	}
	translate +z*1000.000000
}
#declare Layer_ito = union{
	difference{
		intersection{
			plane{ <1.000000,0.000000,0>, 0.500000 }
			plane{ <-1.000000,-0.000000,0>, 0.500000 }
			plane{ <0.000000,1.000000,0>, 0.500000 }
			plane{ <-0.000000,-1.000000,0>, 0.500000 }
			plane{ <1.000000,1.000000,0>, 0.707107 }
			plane{ <-1.000000,-1.000000,0>, 0.707107 }
			plane{ <0,0,-1>, 0 }
			plane{ <0,0,1>, 300.000000 }
		}
// nshapes = 0
		texture { Material_ito }
	}
	translate +z*2000.000000
}
#declare Layers = union {
	//object{ Layer_substrate }
	object{ Layer_nanowire }
	//object{ Layer_ito }
}

Axes
Layers
