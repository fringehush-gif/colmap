/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include "PreProcessor.h"
#include "Reconstructors.h"

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "MyMiscellany.h"
#include "CmdLineParser.h"
#include "PPolynomial.h"
#include "FEMTree.h"
#include "Ply.h"
#include "VertexFactory.h"
#include "RegularGrid.h"
#include "DataStream.imp.h"

#define DEFAULT_DIMENSION 3

using namespace PoissonRecon;

namespace {

const int DefaultSchedule = (int)ThreadPool::Schedule;
const int DefaultChunkSize = (int)ThreadPool::ChunkSize;

// Command-line options belong to one synchronous invocation.
struct Parameters
{
	Parameters( void ) = default;
	Parameters( const Parameters& ) = delete;
	Parameters& operator=( const Parameters& ) = delete;

	CmdLineParameter< char* >
		In{ "in" } ,
		Out{ "out" } ,
		TempDir{ "tempDir" } ,
		Grid{ "grid" } ,
		Tree{ "tree" } ,
		Envelope{ "envelope" } ,
		Transform{ "xForm" };

	CmdLineReadable
		Performance{ "performance" } ,
		ShowResidual{ "showResidual" } ,
		PolygonMesh{ "polygonMesh" } ,
		NonManifold{ "nonManifold" } ,
		ASCII{ "ascii" } ,
		Density{ "density" } ,
		LinearFit{ "linearFit" } ,
		PrimalGrid{ "primalGrid" } ,
		ExactInterpolation{ "exact" } ,
		Colors{ "colors" } ,
		InCore{ "inCore" } ,
		NoDirichletErode{ "noErode" } ,
		Gradients{ "gradients" } ,
		GridCoordinates{ "gridCoordinates" } ,
		Confidence{ "confidence" } ,
		Verbose{ "verbose" };

	CmdLineParameter< int >
#ifndef FAST_COMPILE
		Degree{ "degree" , Reconstructor::Poisson::DefaultFEMDegree } ,
#endif // !FAST_COMPILE
		Depth{ "depth" , 8 } ,
		KernelDepth{ "kernelDepth" , -1 } ,
		SolveDepth{ "solveDepth" , -1 } ,
		EnvelopeDepth{ "envelopeDepth" , -1 } ,
		FullDepth{ "fullDepth" , 5 } ,
		BaseDepth{ "baseDepth" , -1 } ,
		Iters{ "iters" , 8 } ,
		BaseVCycles{ "baseVCycles" , 1 } ,
#ifndef FAST_COMPILE
		BType{ "bType" , Reconstructor::Poisson::DefaultFEMBoundary+1 } ,
#endif // !FAST_COMPILE
		MaxMemoryGB{ "maxMemory" , 0 } ,
		ParallelType{ "parallel" , 0 } ,
		AlignmentDir{ "alignDir" , DEFAULT_DIMENSION-1 } ,
		ScheduleType{ "schedule" , DefaultSchedule } ,
		ThreadChunkSize{ "chunkSize" , DefaultChunkSize };

	CmdLineParameter< float >
		DataX{ "data" , 32.f } ,
		SamplesPerNode{ "samplesPerNode" , 1.5f } ,
		Scale{ "scale" , 1.1f } ,
		Width{ "width" , 0.f } ,
		CGSolverAccuracy{ "cgAccuracy" , 1e-3f } ,
		LowDepthCutOff{ "lowDepthCutOff" , 0.f } ,
		PointWeight{ "pointWeight" };

	std::vector< CmdLineReadable* > params =
	{
#ifndef FAST_COMPILE
		&Degree , &BType ,
#endif // !FAST_COMPILE
		&In , &Depth , &Out , &Transform ,
		&SolveDepth ,
		&Envelope ,
		&Width ,
		&Scale , &Verbose , &CGSolverAccuracy ,
		&KernelDepth , &SamplesPerNode , &Confidence , &NonManifold , &PolygonMesh , &ASCII , &ShowResidual ,
		&EnvelopeDepth ,
		&NoDirichletErode ,
		&BaseDepth , &BaseVCycles ,
		&PointWeight ,
		&Grid ,
		&Tree ,
		&Density ,
		&FullDepth ,
		&Iters ,
		&DataX ,
		&Colors ,
		&Gradients ,
		&LinearFit ,
		&PrimalGrid ,
		&TempDir ,
		&ExactInterpolation ,
		&Performance ,
		&MaxMemoryGB ,
		&InCore ,
		&ParallelType ,
		&ScheduleType ,
		&ThreadChunkSize ,
		&LowDepthCutOff ,
		&AlignmentDir ,
		&GridCoordinates ,
		NULL
	};

};

void ShowUsage( const Parameters &options , char* ex )
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input points>\n" , options.In.name );
	printf( "\t[--%s <input envelope>\n" , options.Envelope.name );
	printf( "\t[--%s <ouput triangle mesh>]\n" , options.Out.name );
	printf( "\t[--%s <ouput grid>]\n" , options.Grid.name );
	printf( "\t[--%s <ouput fem tree>]\n" , options.Tree.name );
#ifndef FAST_COMPILE
	printf( "\t[--%s <b-spline degree>=%d]\n" , options.Degree.name , options.Degree.value );
	printf( "\t[--%s <boundary type>=%d]\n" , options.BType.name , options.BType.value );
	for( int i=0 ; i<BOUNDARY_COUNT ; i++ ) printf( "\t\t%d] %s\n" , i+1 , BoundaryNames[i] );
#endif // !FAST_COMPILE
	printf( "\t[--%s <maximum reconstruction depth>=%d]\n" , options.Depth.name , options.Depth.value );
	printf( "\t[--%s <maximum solution depth>=%d]\n" , options.SolveDepth.name , options.SolveDepth.value );
	printf( "\t[--%s <grid width>]\n" , options.Width.name );
	printf( "\t[--%s <full depth>=%d]\n" , options.FullDepth.name , options.FullDepth.value );
	printf( "\t[--%s <envelope depth>=%d]\n" , options.EnvelopeDepth.name , options.EnvelopeDepth.value );
	printf( "\t[--%s <coarse MG solver depth>]\n" , options.BaseDepth.name );
	printf( "\t[--%s <coarse MG solver v-cycles>=%d]\n" , options.BaseVCycles.name , options.BaseVCycles.value );
	printf( "\t[--%s <scale factor>=%f]\n" , options.Scale.name , options.Scale.value );
	printf( "\t[--%s <minimum number of samples per node>=%f]\n" , options.SamplesPerNode.name, options.SamplesPerNode.value );
	printf( "\t[--%s <interpolation weight>=%.3e * <b-spline degree>]\n" , options.PointWeight.name , Reconstructor::Poisson::WeightMultiplier * Reconstructor::Poisson::DefaultFEMDegree );
	printf( "\t[--%s <iterations>=%d]\n" , options.Iters.name , options.Iters.value );
	printf( "\t[--%s]\n" , options.ExactInterpolation.name );
	printf( "\t[--%s <pull factor>=%f]\n" , options.DataX.name , options.DataX.value );
	printf( "\t[--%s]\n" , options.Colors.name );
	printf( "\t[--%s]\n" , options.Gradients.name );
	printf( "\t[--%s <parallel type>=%d]\n" , options.ParallelType.name , options.ParallelType.value );
	for( size_t i=0 ; i<ThreadPool::ParallelNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ParallelNames[i].c_str() );
	printf( "\t[--%s <schedue type>=%d]\n" , options.ScheduleType.name , options.ScheduleType.value );
	for( size_t i=0 ; i<ThreadPool::ScheduleNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ScheduleNames[i].c_str() );
	printf( "\t[--%s <thread chunk size>=%d]\n" , options.ThreadChunkSize.name , options.ThreadChunkSize.value );
	printf( "\t[--%s <low depth cut-off>=%f]\n" , options.LowDepthCutOff.name , options.LowDepthCutOff.value );
	printf( "\t[--%s <alignment direction>=%d]\n" , options.AlignmentDir.name , options.AlignmentDir.value );
	printf( "\t[--%s]\n" , options.Confidence.name );
	printf( "\t[--%s]\n" , options.NonManifold.name );
	printf( "\t[--%s]\n" , options.PolygonMesh.name );
	printf( "\t[--%s <cg solver accuracy>=%g]\n" , options.CGSolverAccuracy.name , options.CGSolverAccuracy.value );
	printf( "\t[--%s <maximum memory (in GB)>=%d]\n" , options.MaxMemoryGB.name , options.MaxMemoryGB.value );
	printf( "\t[--%s]\n" , options.NoDirichletErode.name );
	printf( "\t[--%s]\n" , options.GridCoordinates.name );
	printf( "\t[--%s]\n" , options.Performance.name );
	printf( "\t[--%s]\n" , options.Density.name );
	printf( "\t[--%s]\n" , options.LinearFit.name );
	printf( "\t[--%s]\n" , options.PrimalGrid.name );
	printf( "\t[--%s]\n" , options.ASCII.name );
	printf( "\t[--%s]\n" , options.TempDir.name );
	printf( "\t[--%s]\n" , options.InCore.name );
	printf( "\t[--%s]\n" , options.Verbose.name );
}

template< typename Real , unsigned int Dim , unsigned int FEMSig , bool HasGradients , bool HasDensity , bool InCore , typename ... AuxDataFactories >
void WriteMesh
(
	Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactories::VertexType ... > &implicit ,
	const Reconstructor::LevelSetExtractionParameters &meParams ,
	std::string fileName ,
	bool ascii ,
	const AuxDataFactories& ... factories
)
{
	// A description of the output vertex information
	using VInfo = Reconstructor::OutputVertexInfo< Real , Dim , HasGradients , HasDensity , AuxDataFactories ... >;

	// A factory generating the output vertices
	using Factory = typename VInfo::Factory;
	Factory factory = VInfo::GetFactory( factories... );

	Reconstructor::OutputInputFactoryTypeStream< Real , Dim , Factory , InCore , true , typename AuxDataFactories::VertexType... > vertexStream( factory , VInfo::Convert );
	Reconstructor::OutputInputFaceStream< Dim-1 , InCore , true > faceStream;

	implicit.extractLevelSet( vertexStream , faceStream , meParams );

	// Write the mesh to a .ply file
	std::vector< std::string > noComments;
	vertexStream.reset();
	PLY::Write< Factory , node_index_type , Real , Dim >( fileName , factory , vertexStream.size() , faceStream.size() , vertexStream , faceStream , ascii ? PLY_ASCII : PLY_BINARY_NATIVE , noComments );
}

template< typename Real , unsigned int Dim , unsigned int FEMSig , bool HasDensity , bool InCore , typename ... AuxDataFactories >
void WriteMesh
(
	bool hasGradients ,
	Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactories::VertexType ... > &implicit ,
	const Reconstructor::LevelSetExtractionParameters &meParams ,
	std::string fileName ,
	bool ascii ,
	const AuxDataFactories& ... factories
)
{
	if( hasGradients ) return WriteMesh< Real , Dim, FEMSig , true  , HasDensity , InCore , AuxDataFactories ... >( implicit , meParams , fileName , ascii , factories... );
	else               return WriteMesh< Real , Dim, FEMSig , false , HasDensity , InCore , AuxDataFactories ... >( implicit , meParams , fileName , ascii , factories... );
}

template< typename Real , unsigned int Dim , unsigned int FEMSig , bool InCore , typename ... AuxDataFactories >
void WriteMesh
(
	bool hasGradients , bool hasDensity ,
	Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactories::VertexType ... > &implicit ,
	const Reconstructor::LevelSetExtractionParameters &meParams ,
	std::string fileName ,
	bool ascii ,
	const AuxDataFactories& ... factories
)
{
	if( hasDensity ) return WriteMesh< Real , Dim, FEMSig , true  , InCore , AuxDataFactories ... >( hasGradients , implicit , meParams , fileName , ascii , factories... );
	else             return WriteMesh< Real , Dim, FEMSig , false , InCore , AuxDataFactories ... >( hasGradients , implicit , meParams , fileName , ascii , factories... );
}

template< typename Real , unsigned int Dim , unsigned int FEMSig , typename ... AuxDataFactories >
void WriteMesh
(
	bool hasGradients , bool hasDensity , bool inCore ,
	Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactories::VertexType ... > &implicit ,
	const Reconstructor::LevelSetExtractionParameters &meParams ,
	std::string fileName ,
	bool ascii ,
	const AuxDataFactories& ... factories
)
{
	if( inCore ) return WriteMesh< Real , Dim, FEMSig , true  , AuxDataFactories ... >( hasGradients , hasDensity , implicit , meParams , fileName , ascii , factories... );
	else         return WriteMesh< Real , Dim, FEMSig , false , AuxDataFactories ... >( hasGradients , hasDensity , implicit , meParams , fileName , ascii , factories... );
}

template< class Real , unsigned int Dim , unsigned int FEMSig , typename AuxDataFactory >
void Execute( const Parameters &options , const AuxDataFactory &auxDataFactory )
{
	static const bool HasAuxData = !std::is_same< AuxDataFactory , VertexFactory::EmptyFactory< Real > >::value;

	///////////////
	// Types --> //
	typedef IsotropicUIntPack< Dim , FEMSig > Sigs;
	using namespace VertexFactory;

	// The factory for constructing an input sample's data
	typedef std::conditional_t< HasAuxData , Factory< Real , NormalFactory< Real , Dim > , AuxDataFactory > , NormalFactory< Real , Dim > > InputSampleDataFactory;

	// The factory for constructing an input sample
	typedef Factory< Real , PositionFactory< Real , Dim > , InputSampleDataFactory >  InputSampleFactory;

	typedef InputDataStream< typename InputSampleFactory::VertexType > InputPointStream;

	using Implicit = std::conditional_t< HasAuxData , Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactory::VertexType > , Reconstructor::Implicit< Real , Dim , IsotropicUIntPack< Dim , FEMSig > > >;
	using Solver = std::conditional_t< HasAuxData , Reconstructor::Poisson::Solver< Real , Dim , IsotropicUIntPack< Dim , FEMSig > , typename AuxDataFactory::VertexType > , Reconstructor::Poisson::Solver< Real , Dim , IsotropicUIntPack< Dim , FEMSig > > >;
	// <-- Types //
	///////////////

	if( options.Verbose.set )
	{
		std::cout << "*************************************************************" << std::endl;
		std::cout << "*************************************************************" << std::endl;
		std::cout << "** Running Screened Poisson Reconstruction (Version " << ADAPTIVE_SOLVERS_VERSION << ") **" << std::endl;
		std::cout << "*************************************************************" << std::endl;
		std::cout << "*************************************************************" << std::endl;

		char str[1024];
		for( int i=0 ; options.params[i] ; i++ ) if( options.params[i]->set )
		{
			options.params[i]->writeValue( str );
			if( strlen( str ) ) std::cout << "\t--" << options.params[i]->name << " " << str << std::endl;
			else                std::cout << "\t--" << options.params[i]->name << std::endl;
		}
	}

	Profiler profiler(20);
	std::unique_ptr< Implicit > implicit;
	typename Reconstructor::Poisson::SolutionParameters< Real > sParams;
	Reconstructor::LevelSetExtractionParameters meParams;

	sParams.verbose = options.Verbose.set;
	sParams.dirichletErode = !options.NoDirichletErode.set;
	sParams.exactInterpolation = options.ExactInterpolation.set;
	sParams.showResidual = options.ShowResidual.set;
	sParams.confidence = options.Confidence.set;
	sParams.scale = (Real)options.Scale.value;
	sParams.lowDepthCutOff = (Real)options.LowDepthCutOff.value;
	sParams.width = (Real)options.Width.value;
	sParams.pointWeight = (Real)options.PointWeight.value;
	sParams.samplesPerNode = (Real)options.SamplesPerNode.value;
	sParams.cgSolverAccuracy = (Real)options.CGSolverAccuracy.value;
	sParams.perLevelDataScaleFactor = (Real)options.DataX.value;
	sParams.depth = (unsigned int)options.Depth.value;
	sParams.baseDepth = (unsigned int)options.BaseDepth.value;
	sParams.solveDepth = (unsigned int)options.SolveDepth.value;
	sParams.fullDepth = (unsigned int)options.FullDepth.value;
	sParams.kernelDepth = (unsigned int)options.KernelDepth.value;
	sParams.envelopeDepth = (unsigned int)options.EnvelopeDepth.value;
	sParams.baseVCycles = (unsigned int)options.BaseVCycles.value;
	sParams.iters = (unsigned int)options.Iters.value;
	sParams.alignDir = (unsigned int)options.AlignmentDir.value;

	meParams.linearFit = options.LinearFit.set;
	meParams.outputGradients = options.Gradients.set;
	meParams.forceManifold = !options.NonManifold.set;
	meParams.polygonMesh = options.PolygonMesh.set;
	meParams.gridCoordinates = options.GridCoordinates.set;
	meParams.outputDensity = options.Density.set;
	meParams.verbose = options.Verbose.set;

	double startTime = Time();

	InputSampleFactory *_inputSampleFactory;
	if constexpr( HasAuxData ) _inputSampleFactory = new InputSampleFactory( VertexFactory::PositionFactory< Real , Dim >() , InputSampleDataFactory( VertexFactory::NormalFactory< Real , Dim >() , auxDataFactory ) );
	else _inputSampleFactory = new InputSampleFactory( VertexFactory::PositionFactory< Real , Dim >() , VertexFactory::NormalFactory< Real , Dim >() );
	InputSampleFactory &inputSampleFactory = *_inputSampleFactory;
	XForm< Real , Dim+1 > toModel = XForm< Real , Dim+1 >::Identity();

	// Read in the transform, if we want to apply one to the points before processing
	if( options.Transform.set )
	{
		FILE* fp = fopen( options.Transform.value , "r" );
		if( !fp ) MK_WARN( "Could not read x-form from: " , options.Transform.value );
		else
		{
			for( int i=0 ; i<Dim+1 ; i++ ) for( int j=0 ; j<Dim+1 ; j++ )
			{
				float f;
				if( fscanf( fp , " %f " , &f )!=1 ) MK_THROW( "Failed to read xform" );
				toModel(i,j) = (Real)f;
			}
			fclose( fp );
		}
	}
	std::vector< typename InputSampleFactory::VertexType > inCorePoints;
	InputPointStream *pointStream;

	// Get the point stream
	{
		profiler.reset();
		char *ext = GetFileExtension( options.In.value );

		if( options.InCore.set )
		{
			InputPointStream *_pointStream;
			if     ( !strcasecmp( ext , "bnpts" ) ) _pointStream = new BinaryInputDataStream< InputSampleFactory >( options.In.value , inputSampleFactory );
			else if( !strcasecmp( ext , "ply"   ) ) _pointStream = new    PLYInputDataStream< InputSampleFactory >( options.In.value , inputSampleFactory );
			else                                    _pointStream = new  ASCIIInputDataStream< InputSampleFactory >( options.In.value , inputSampleFactory );
			typename InputSampleFactory::VertexType p = inputSampleFactory();
			while( _pointStream->read( p ) ) inCorePoints.push_back( p );
			delete _pointStream;

			pointStream = new VectorBackedInputDataStream< typename InputSampleFactory::VertexType >( inCorePoints );
		}
		else
		{
			if     ( !strcasecmp( ext , "bnpts" ) ) pointStream = new BinaryInputDataStream< InputSampleFactory >( options.In.value , inputSampleFactory );
			else if( !strcasecmp( ext , "ply"   ) ) pointStream = new    PLYInputDataStream< InputSampleFactory >( options.In.value , inputSampleFactory );
			else                                    pointStream = new  ASCIIInputDataStream< InputSampleFactory >( options.In.value , inputSampleFactory );
		}
		delete[] ext;
	}

	typename Reconstructor::Poisson::EnvelopeMesh< Real , Dim > *envelopeMesh = NULL;
	if( options.Envelope.set )
	{
		envelopeMesh = new typename Reconstructor::Poisson::EnvelopeMesh< Real , Dim >();
		{
			std::vector< std::vector< int > > polygons;
			std::vector< std::string > comments;
			int file_type;
			PLY::ReadPolygons( options.Envelope.value , PositionFactory< Real , Dim >() , envelopeMesh->vertices , polygons , file_type , comments );
			envelopeMesh->simplices.resize( polygons.size() );
			for( int i=0 ; i<polygons.size() ; i++ )
				if( polygons[i].size()!=Dim ) MK_THROW( "Not a simplex" );
				else for( int j=0 ; j<Dim ; j++ ) envelopeMesh->simplices[i][j] = polygons[i][j];
		}
	}

	// A wrapper class to realize InputDataStream< SampleType > as an InputSampleStream
	struct _InputOrientedSampleStream : public Reconstructor::InputOrientedSampleStream< Real , Dim >
	{
		typedef Reconstructor::Normal< Real , Dim > DataType;
		typedef DirectSum< Real , Reconstructor::Position< Real , Dim > , DataType > SampleType;
		typedef InputDataStream< SampleType > _InputPointStream;
		_InputPointStream &pointStream;
		SampleType scratch;

		_InputOrientedSampleStream( _InputPointStream &pointStream )
			: pointStream( pointStream ) , scratch( Reconstructor::Position< Real , Dim >() , Reconstructor::Normal< Real , Dim >() )
		{}

		void reset( void ){ pointStream.reset(); }

		bool read( Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n )
		{
			bool ret = pointStream.read( scratch );
			if( ret ) p = scratch.template get<0>() , n = scratch.template get<1>();
			return ret;
		}

		bool read( unsigned int thread , Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n )
		{
			bool ret = pointStream.read( thread , scratch );
			if( ret ) p = scratch.template get<0>() , n = scratch.template get<1>();
			return ret;
		}
	};

	// A wrapper class to realize InputDataStream< SampleType > as an InputSampleWithDataStream
	struct _InputOrientedSampleWithDataStream : public Reconstructor::InputOrientedSampleStream< Real , Dim , typename AuxDataFactory::VertexType >
	{
		typedef DirectSum< Real , Reconstructor::Normal< Real , Dim > , typename AuxDataFactory::VertexType > DataType;
		typedef DirectSum< Real , Reconstructor::Position< Real , Dim > , DataType > SampleType;
		typedef InputDataStream< SampleType > _InputPointStream;
		_InputPointStream &pointStream;
		SampleType scratch;

		_InputOrientedSampleWithDataStream( _InputPointStream &pointStream , typename AuxDataFactory::VertexType zero )
			: pointStream( pointStream ) , scratch( Reconstructor::Position< Real , Dim >() , DataType( Reconstructor::Normal< Real , Dim >() , zero ) )
		{}

		void reset( void ){ pointStream.reset(); }

		bool read( Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n , typename AuxDataFactory::VertexType &d )
		{
			bool ret = pointStream.read( scratch );
			if( ret ) p = scratch.template get<0>() , n = scratch.template get<1>().template get<0>() , d = scratch.template get<1>().template get<1>();
			return ret;
		}

		bool read( unsigned int thread , Reconstructor::Position< Real , Dim > &p , Reconstructor::Normal< Real , Dim > &n , typename AuxDataFactory::VertexType &d )
		{
			bool ret = pointStream.read( thread , scratch );
			if( ret ) p = scratch.template get<0>() , n = scratch.template get<1>().template get<0>() , d = scratch.template get<1>().template get<1>();
			return ret;
		}
	};

	if( options.Transform.set && envelopeMesh ) for( unsigned int i=0 ; i<envelopeMesh->vertices.size() ; i++ ) envelopeMesh->vertices[i] = toModel * envelopeMesh->vertices[i];
	if constexpr( HasAuxData )
	{
		_InputOrientedSampleWithDataStream sampleStream( *pointStream , auxDataFactory() );

		if( options.Transform.set )
		{
			Reconstructor::TransformedInputOrientedSampleStream< Real , Dim , typename AuxDataFactory::VertexType > _sampleStream( toModel , sampleStream );
			implicit.reset( Solver::Solve( _sampleStream , sParams , auxDataFactory() , envelopeMesh ) );
			implicit->unitCubeToModel = toModel.inverse() * implicit->unitCubeToModel;
		}
		else implicit.reset( Solver::Solve( sampleStream , sParams , auxDataFactory() , envelopeMesh ) );
	}
	else
	{
		_InputOrientedSampleStream sampleStream( *pointStream );

		if( options.Transform.set )
		{
			Reconstructor::TransformedInputOrientedSampleStream< Real , Dim > _sampleStream( toModel , sampleStream );
			implicit.reset( Solver::Solve( _sampleStream , sParams , envelopeMesh ) );
			implicit->unitCubeToModel = toModel.inverse() * implicit->unitCubeToModel;
		}
		else implicit.reset( Solver::Solve( sampleStream , sParams , envelopeMesh ) );
	}

	delete pointStream;
	delete _inputSampleFactory;
	delete envelopeMesh;

	if( options.Tree.set )
	{
		FILE* fp = fopen( options.Tree.value , "wb" );
		if( !fp ) MK_THROW( "Failed to open file for writing: " , options.Tree.value );
		FileStream fs(fp);
		FEMTree< Dim , Real >::WriteParameter( fs );
		DenseNodeData< Real , Sigs >::WriteSignatures( fs );
		implicit->tree.write( fs , false );
		fs.write( implicit->unitCubeToModel.inverse() );
		implicit->solution.write( fs );
		fclose( fp );
	}

	if( options.Grid.set )
	{
		int res = 0;
		profiler.reset();
		Pointer( Real ) values = implicit->tree.template regularGridEvaluate< true >( implicit->solution , res , -1 , options.PrimalGrid.set );
		if( options.Verbose.set ) std::cout << "Got grid: " << profiler << std::endl;
		XForm< Real , Dim+1 > voxelToUnitCube = XForm< Real , Dim+1 >::Identity();
		if( options.PrimalGrid.set ) for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / (res-1) );
		else                 for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / res ) , voxelToUnitCube( Dim , d ) = (Real)( 0.5 / res );

		unsigned int _res[Dim];
		for( int d=0 ; d<Dim ; d++ ) _res[d] = res;
		RegularGrid< Real , Dim >::Write( options.Grid.value , _res , values , implicit->unitCubeToModel * voxelToUnitCube );

		DeletePointer( values );
	}


	if( options.Out.set )
		if( Dim==2 || Dim==3 )
		{
			// Create the output mesh
			char tempHeader[2048];
			{
				char tempPath[1024];
				tempPath[0] = 0;
				if( options.TempDir.set ) strcpy( tempPath , options.TempDir.value );
				else SetTempDirectory( tempPath , sizeof(tempPath) );
				if( strlen(tempPath)==0 ) sprintf( tempPath , ".%c" , FileSeparator );
				if( tempPath[ strlen( tempPath )-1 ]==FileSeparator ) sprintf( tempHeader , "%sPR_" , tempPath );
				else                                                  sprintf( tempHeader , "%s%cPR_" , tempPath , FileSeparator );
			}

			XForm< Real , Dim+1 > pXForm = implicit->unitCubeToModel;
			XForm< Real , Dim > nXForm = XForm< Real , Dim >( pXForm ).inverse().transpose();

			if constexpr( HasAuxData ) WriteMesh< Real , Dim , FEMSig >( options.Gradients.set , options.Density.set , options.InCore.set , *implicit , meParams , options.Out.value , options.ASCII.set , auxDataFactory );
			else                       WriteMesh< Real , Dim , FEMSig >( options.Gradients.set , options.Density.set , options.InCore.set , *implicit , meParams , options.Out.value , options.ASCII.set );
		}
		else MK_WARN( "Mesh extraction is only supported in dimensions 2 and 3" );

	if( options.Verbose.set ) std::cout << "#          Total Solve: " << Time()-startTime << " (s), " << MemoryInfo::PeakMemoryUsageMB() << " (MB)" << std::endl;
}

#ifndef FAST_COMPILE
template< unsigned int Dim , class Real , BoundaryType BType , typename AuxDataFactory >
void Execute( const Parameters &options , const AuxDataFactory &auxDataFactory )
{
	switch( options.Degree.value )
	{
		case 1: return Execute< Real , Dim , FEMDegreeAndBType< 1 , BType >::Signature >( options , auxDataFactory );
		case 2: return Execute< Real , Dim , FEMDegreeAndBType< 2 , BType >::Signature >( options , auxDataFactory );
//		case 3: return Execute< Real , Dim , FEMDegreeAndBType< 3 , BType >::Signature >( options , auxDataFactory );
//		case 4: return Execute< Real , Dim , FEMDegreeAndBType< 4 , BType >::Signature >( options , auxDataFactory );
		default: MK_THROW( "Only B-Splines of degree 1 - 2 are supported" );
	}
}

template< unsigned int Dim , class Real , typename AuxDataFactory >
void Execute( const Parameters &options , const AuxDataFactory &auxDataFactory )
{
	switch( options.BType.value )
	{
		case BOUNDARY_FREE+1:      return Execute< Dim , Real , BOUNDARY_FREE      >( options , auxDataFactory );
		case BOUNDARY_NEUMANN+1:   return Execute< Dim , Real , BOUNDARY_NEUMANN   >( options , auxDataFactory );
		case BOUNDARY_DIRICHLET+1: return Execute< Dim , Real , BOUNDARY_DIRICHLET >( options , auxDataFactory );
		default: MK_THROW( "Not a valid boundary type: " , options.BType.value );
	}
}
#endif // !FAST_COMPILE

} // namespace

int RunPoissonRecon( int argc , char* argv[] )
{
	Timer timer;
#ifdef ARRAY_DEBUG
	MK_WARN( "Array debugging enabled" );
#endif // ARRAY_DEBUG
	Parameters options;
	CmdLineParse( argc-1 , &argv[1] , options.params.data() );
	if( options.MaxMemoryGB.value>0 ) SetPeakMemoryMB( options.MaxMemoryGB.value<<10 );
	ThreadPool::ChunkSize = options.ThreadChunkSize.value;
	ThreadPool::Schedule = (ThreadPool::ScheduleType)options.ScheduleType.value;
	ThreadPool::ParallelizationType= (ThreadPool::ParallelType)options.ParallelType.value;

	if( !options.In.set )
	{
		ShowUsage( options , argv[0] );
		return 0;
	}

#ifdef USE_DOUBLE
	typedef double Real;
#else // !USE_DOUBLE
	typedef float  Real;
#endif // USE_DOUBLE

#ifdef FAST_COMPILE
	static const int Degree = Reconstructor::Poisson::DefaultFEMDegree;
	static const BoundaryType BType = Reconstructor::Poisson::DefaultFEMBoundary;
	static const unsigned int Dim = DEFAULT_DIMENSION;
	static const unsigned int FEMSig = FEMDegreeAndBType< Degree , BType >::Signature;
	MK_WARN( "Compiled for degree-" , Degree , ", boundary-" , BoundaryNames[ BType ] , ", " , sizeof(Real)==4 ? "single" : "double" , "-precision _only_" );
	if( !options.PointWeight.set ) options.PointWeight.value = Reconstructor::Poisson::WeightMultiplier*Degree;
	char *ext = GetFileExtension( options.In.value );
	if( !strcasecmp( ext , "ply" ) )
	{
		typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , DEFAULT_DIMENSION > , typename VertexFactory::NormalFactory< Real , DEFAULT_DIMENSION > > Factory;
		Factory factory;
		bool *readFlags = new bool[ factory.plyReadNum() ];
		std::vector< PlyProperty > unprocessedProperties;
		PLY::ReadVertexHeader( options.In.value , factory , readFlags , unprocessedProperties );
		if( !factory.template plyValidReadProperties<0>( readFlags ) ) MK_THROW( "Ply file does not contain positions" );
		if( !factory.template plyValidReadProperties<1>( readFlags ) ) MK_THROW( "Ply file does not contain normals" );
		delete[] readFlags;
#if defined(POISSON_AUX_NATIVE_GPU)
        const bool standard_rgb = unprocessedProperties.size()==3 &&
            unprocessedProperties[0].name=="red" &&
            unprocessedProperties[1].name=="green" &&
            unprocessedProperties[2].name=="blue" &&
            unprocessedProperties[0].external_type==PLY_UCHAR &&
            unprocessedProperties[1].external_type==PLY_UCHAR &&
            unprocessedProperties[2].external_type==PLY_UCHAR &&
            !unprocessedProperties[0].is_list &&
            !unprocessedProperties[1].is_list &&
            !unprocessedProperties[2].is_list;
        const ::AuxNativeInput::Scope rgb_input_scope(standard_rgb);
#endif

#if defined(POISSON_AUX_NATIVE_GPU)
		if( standard_rgb ) Execute< Real , Dim , FEMSig >( options , VertexFactory::RGBColorFactory< Real >() );
		else
#endif
		if( unprocessedProperties.size() ) Execute< Real , Dim , FEMSig >( options , VertexFactory::DynamicFactory< Real >( unprocessedProperties ) );
		else                               Execute< Real , Dim , FEMSig >( options , VertexFactory::EmptyFactory< Real >() );
	}
	else
	{
		if( options.Colors.set ) Execute< Real , Dim , FEMSig >( options , VertexFactory::RGBColorFactory< Real >() );
		else             Execute< Real , Dim , FEMSig >( options , VertexFactory::EmptyFactory< Real >() );
	}
	delete[] ext;
#else // !FAST_COMPILE
	if( !options.PointWeight.set ) options.PointWeight.value = Reconstructor::Poisson::WeightMultiplier * options.Degree.value;
	char *ext = GetFileExtension( options.In.value );
	if( !strcasecmp( ext , "ply" ) )
	{
		typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , DEFAULT_DIMENSION > , typename VertexFactory::NormalFactory< Real , DEFAULT_DIMENSION > > Factory;
		Factory factory;
		bool *readFlags = new bool[ factory.plyReadNum() ];
		std::vector< PlyProperty > unprocessedProperties;
		PLY::ReadVertexHeader( options.In.value , factory , readFlags , unprocessedProperties );
		if( !factory.template plyValidReadProperties<0>( readFlags ) ) MK_THROW( "Ply file does not contain positions" );
		if( !factory.template plyValidReadProperties<1>( readFlags ) ) MK_THROW( "Ply file does not contain normals" );
		delete[] readFlags;
#if defined(POISSON_AUX_NATIVE_GPU)
        const bool standard_rgb = unprocessedProperties.size()==3 &&
            unprocessedProperties[0].name=="red" &&
            unprocessedProperties[1].name=="green" &&
            unprocessedProperties[2].name=="blue" &&
            unprocessedProperties[0].external_type==PLY_UCHAR &&
            unprocessedProperties[1].external_type==PLY_UCHAR &&
            unprocessedProperties[2].external_type==PLY_UCHAR &&
            !unprocessedProperties[0].is_list &&
            !unprocessedProperties[1].is_list &&
            !unprocessedProperties[2].is_list;
        const ::AuxNativeInput::Scope rgb_input_scope(standard_rgb);
#endif

#if defined(POISSON_AUX_NATIVE_GPU)
		if( standard_rgb ) Execute< DEFAULT_DIMENSION , Real >( options , VertexFactory::RGBColorFactory< Real >() );
		else
#endif
		if( unprocessedProperties.size() ) Execute< DEFAULT_DIMENSION , Real >( options , VertexFactory::DynamicFactory< Real >( unprocessedProperties ) );
		else                               Execute< DEFAULT_DIMENSION , Real >( options , VertexFactory::EmptyFactory< Real >() );
	}
	else
	{
		if( options.Colors.set ) Execute< DEFAULT_DIMENSION , Real >( options , VertexFactory::RGBColorFactory< Real >() );
		else             Execute< DEFAULT_DIMENSION , Real >( options , VertexFactory::EmptyFactory< Real >() );
	}
	delete[] ext;
#endif // FAST_COMPILE
	if( options.Performance.set )
	{
		printf( "Time (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
		printf( "Peak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );
	}

	return EXIT_SUCCESS;
}
