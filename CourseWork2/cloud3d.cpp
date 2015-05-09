#include "cloud3d.h"


cloud3d::cloud3d()
{
}

cloud3d::~cloud3d()
{
}

int cloud3d::addPoint(Point3d point)
{
	vertexes.push_back(point);
	return vertexes.size() - 1;
}

void cloud3d::dumpPLY(std::ostream& out)
{
	out << "ply" << std::endl;
	out << "format ascii 1.0" << std::endl;
	out << "comment made by ViMouse software" << std::endl;
	out << "comment This file is a saved stereo-reconstruction" << std::endl;
	out << "element vertex " << vertexes.size() << std::endl;
	out << "property float x" << std::endl;
	out << "property float y" << std::endl;
	out << "property float z" << std::endl;
	out << "property uchar red" << std::endl;
	out << "property uchar green" << std::endl;
	out << "property uchar blue" << std::endl;
	out << "end_header" << std::endl;

	for (unsigned i = 0; i < vertexes.size(); i++)
	{
		out << vertexes[i].x << " "
			<< vertexes[i].y << " "
			<< vertexes[i].z << " ";
			out << (unsigned)(128) << " "
				<< (unsigned)(128) << " "
				<< (unsigned)(128) << std::endl;
	}
	//    SYNC_PRINT(("This 0x%X. Edges %d", this, edges.size()));

}